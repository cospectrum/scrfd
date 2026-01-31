import * as ort from 'onnxruntime-web';
import { Bbox } from './types';

// Point ONNX Runtime to WASM files (Vite doesn't serve them from node_modules)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
import type { Face, FaceKeypoints, Point, Threshold } from './types';

const KPS = 5;
const IH = 640;
const IW = 640;
const CH = 3;

interface Options {
  fmc: number;
  numAnchors: number;
  strides: [number, number, number];
}

function numOfOutputs(options: Options): number {
  return options.strides.length + 2 * options.fmc;
}

interface RawFaces {
  scores: Float32Array;
  boxes: Float32Array;
  keypoints: Float32Array;
  scoresShape: [number, number];
  boxesShape: [number, number];
  keypointsShape: [number, number];
}

interface AnchorOptions {
  width: number;
  height: number;
  scale: number;
  repeat: number;
}

export class Scrfd {
  private session: ort.InferenceSession | null = null;
  private options: Options;

  private constructor(session: ort.InferenceSession, options: Options) {
    this.session = session;
    this.options = options;
  }

  static async fromBytes(modelBytes: ArrayBuffer): Promise<Scrfd> {
    const session = await Scrfd.createSessionWithBackendFallback(modelBytes);
    const numOutputs = session.outputNames.length;
    const options = Scrfd.determineOptions(numOutputs);
    
    if (numOutputs !== numOfOutputs(options)) {
      throw new Error(`Mismatch: expected ${numOfOutputs(options)} outputs, got ${numOutputs}`);
    }

    return new Scrfd(session, options);
  }

  static async fromUrl(url: string): Promise<Scrfd> {
    const response = await fetch(url);
    const modelBytes = await response.arrayBuffer();
    return Scrfd.fromBytes(modelBytes);
  }

  private static async createSessionWithBackendFallback(
    modelBytes: ArrayBuffer
  ): Promise<ort.InferenceSession> {
    const backends: string[] = ['webnn', 'webgpu', 'webgl', 'wasm', 'cpu'];
    let lastError: Error | null = null;

    for (const backend of backends) {
      try {
        const session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: [backend],
        });
        console.log(`Successfully initialized with backend: ${backend}`);
        return session;
      } catch (error) {
        console.warn(`Backend ${backend} not available:`, error);
        lastError = error as Error;
        continue;
      }
    }

    throw new Error(
      `No suitable backend found. Tried: ${backends.join(', ')}. Last error: ${lastError?.message}`
    );
  }

  private static determineOptions(numOutputs: number): Options {
    switch (numOutputs) {
      case 9:
        return {
          fmc: 3,
          numAnchors: 2,
          strides: [8, 16, 32],
        };
      default:
        throw new Error(`Unsupported SCRFD architecture with ${numOutputs} outputs`);
    }
  }

  detect(img: ImageData | HTMLImageElement | HTMLCanvasElement): Promise<Face[]> {
    const threshold: Threshold = {
      score: 0.4,
      iou: 0.5,
    };
    return this.detectWithThreshold(img, threshold);
  }

  async detectWithThreshold(
    img: ImageData | HTMLImageElement | HTMLCanvasElement,
    threshold: Threshold
  ): Promise<Face[]> {
    if (threshold.score < 0 || threshold.score > 1) {
      throw new Error('threshold.score must be between 0 and 1');
    }
    if (threshold.iou < 0 || threshold.iou > 1) {
      throw new Error('threshold.iou must be between 0 and 1');
    }

    const { scale, blob } = preprocess(img);
    const rawFaces = await this.forward(blob, threshold.score);
    return postprocess(rawFaces, scale, threshold);
  }

  private async forward(blob: Float32Array, scoreThreshold: number): Promise<RawFaces> {
    if (scoreThreshold < 0 || scoreThreshold > 1) {
      throw new Error('scoreThreshold must be between 0 and 1');
    }

    if (!this.session) {
      throw new Error('Session not initialized');
    }

    // Create tensor from blob: shape [1, CH, IH, IW]
    const inputTensor = new ort.Tensor('float32', blob, [1, CH, IH, IW]);
    const inputName = this.session.inputNames[0];
    const feeds = { [inputName]: inputTensor };

    const results = await this.session.run(feeds);
    const outputNames = this.session.outputNames;

    if (outputNames.length !== numOfOutputs(this.options)) {
      throw new Error(
        `Expected ${numOfOutputs(this.options)} outputs, got ${outputNames.length}`
      );
    }

    const facesBatches: RawFaces[] = [];

    for (let idx = 0; idx < this.options.strides.length; idx++) {
      const stride = this.options.strides[idx];
      
      if (IW % stride !== 0 || IH % stride !== 0) {
        throw new Error(`Stride ${stride} must divide IW (${IW}) and IH (${IH})`);
      }

      const anchorOptions: AnchorOptions = {
        width: IW / stride,
        height: IH / stride,
        repeat: this.options.numAnchors,
        scale: stride,
      };

      const anchorCenters = createAnchors(anchorOptions);
      const n = anchorCenters.length / 2;

      // Get outputs
      const scoresTensor = results[outputNames[idx]];
      const boxesTensor = results[outputNames[idx + this.options.fmc]];
      const keypointsTensor = results[outputNames[idx + 2 * this.options.fmc]];

      let scores = new Float32Array(scoresTensor.data as ArrayLike<number>);
      let boxes = new Float32Array(boxesTensor.data as ArrayLike<number>);
      let keypoints = new Float32Array(keypointsTensor.data as ArrayLike<number>);

      // Handle shape variations: [n, ...] or [1, n, ...]
      const scoresShape = scoresTensor.dims;
      const boxesShape = boxesTensor.dims;
      const keypointsShape = keypointsTensor.dims;

      if (
        !(
          (scoresShape.length === 2 && scoresShape[0] === n && scoresShape[1] === 1) ||
          (scoresShape.length === 3 && scoresShape[0] === 1 && scoresShape[1] === n && scoresShape[2] === 1)
        )
      ) {
        throw new Error(`Unexpected scores shape: ${scoresShape.join(',')}`);
      }

      if (
        !(
          (boxesShape.length === 2 && boxesShape[0] === n && boxesShape[1] === 4) ||
          (boxesShape.length === 3 && boxesShape[0] === 1 && boxesShape[1] === n && boxesShape[2] === 4)
        )
      ) {
        throw new Error(`Unexpected boxes shape: ${boxesShape.join(',')}`);
      }

      if (
        !(
          (keypointsShape.length === 2 && keypointsShape[0] === n && keypointsShape[1] === 2 * KPS) ||
          (keypointsShape.length === 3 && keypointsShape[0] === 1 && keypointsShape[1] === n && keypointsShape[2] === 2 * KPS)
        )
      ) {
        throw new Error(`Unexpected keypoints shape: ${keypointsShape.join(',')}`);
      }

      // Reshape if needed (remove batch dimension - take first batch)
      if (scoresShape.length === 3) {
        scores = scores.slice(0, scoresShape[1] * scoresShape[2]);
      }
      if (boxesShape.length === 3) {
        boxes = boxes.slice(0, boxesShape[1] * boxesShape[2]);
      }
      if (keypointsShape.length === 3) {
        keypoints = keypoints.slice(0, keypointsShape[1] * keypointsShape[2]);
      }

      // Scale by stride
      boxes = boxes.map((v) => v * stride);
      keypoints = keypoints.map((v) => v * stride);

      // Convert to 2D arrays for easier manipulation
      const scores2d = array1dTo2d(scores, n, 1);
      const boxes2d = array1dTo2d(boxes, n, 4);
      const keypoints2d = array1dTo2d(keypoints, n, 2 * KPS);

      // Transform distances to boxes and keypoints
      const boxesTransformed = distance2boxes(anchorCenters, boxes2d);
      const keypointsTransformed = distance2kps(anchorCenters, keypoints2d);

      // Filter by score threshold
      const likely: number[] = [];
      for (let i = 0; i < n; i++) {
        if (scores2d[i][0] >= scoreThreshold) {
          likely.push(i);
        }
      }

      const filteredScores = take(scores2d, likely);
      const filteredBoxes = take(boxesTransformed, likely);
      const filteredKeypoints = take(keypointsTransformed, likely);

      facesBatches.push({
        scores: array2dTo1d(filteredScores),
        boxes: array2dTo1d(filteredBoxes),
        keypoints: array2dTo1d(filteredKeypoints),
        scoresShape: [filteredScores.length, 1],
        boxesShape: [filteredBoxes.length, 4],
        keypointsShape: [filteredKeypoints.length, 2 * KPS],
      });
    }

    // Concatenate all batches
    const allScores: number[][] = [];
    const allBoxes: number[][] = [];
    const allKeypoints: number[][] = [];

    for (const batch of facesBatches) {
      allScores.push(...array1dTo2d(batch.scores, batch.scoresShape[0], batch.scoresShape[1]));
      allBoxes.push(...array1dTo2d(batch.boxes, batch.boxesShape[0], batch.boxesShape[1]));
      allKeypoints.push(
        ...array1dTo2d(batch.keypoints, batch.keypointsShape[0], batch.keypointsShape[1])
      );
    }

    return {
      scores: array2dTo1d(allScores),
      boxes: array2dTo1d(allBoxes),
      keypoints: array2dTo1d(allKeypoints),
      scoresShape: [allScores.length, 1],
      boxesShape: [allBoxes.length, 4],
      keypointsShape: [allKeypoints.length, 2 * KPS],
    };
  }
}

function createAnchors(options: AnchorOptions): Float32Array {
  if (options.repeat <= 0 || options.width <= 0 || options.height <= 0) {
    throw new Error('Invalid anchor options');
  }

  const numOfPoints = options.repeat * options.width * options.height;
  const buf = new Float32Array(numOfPoints * 2);

  let idx = 0;
  for (let y = 0; y < options.height; y++) {
    for (let x = 0; x < options.width; x++) {
      for (let r = 0; r < options.repeat; r++) {
        buf[idx++] = options.scale * x;
        buf[idx++] = options.scale * y;
      }
    }
  }

  return buf;
}

function distance2boxes(points: Float32Array, distance: number[][]): number[][] {
  const n = points.length / 2;
  if (distance.length !== n || distance[0].length !== 4) {
    throw new Error('Invalid shapes for distance2boxes');
  }

  const boxes: number[][] = [];
  for (let i = 0; i < n; i++) {
    const px = points[i * 2];
    const py = points[i * 2 + 1];
    const [d0, d1, d2, d3] = distance[i];
    boxes.push([px - d0, py - d1, px + d2, py + d3]);
  }
  return boxes;
}

function distance2kps(points: Float32Array, distance: number[][]): number[][] {
  const n = points.length / 2;
  if (distance.length !== n || distance[0].length !== 2 * KPS) {
    throw new Error('Invalid shapes for distance2kps');
  }

  const kps: number[][] = [];
  for (let i = 0; i < n; i++) {
    const px = points[i * 2];
    const py = points[i * 2 + 1];
    const kpRow: number[] = [];
    for (let j = 0; j < 2 * KPS; j += 2) {
      kpRow.push(px + distance[i][j]);
      kpRow.push(py + distance[i][j + 1]);
    }
    kps.push(kpRow);
  }
  return kps;
}

function take<T>(array: T[][], indices: number[]): T[][] {
  return indices.map((idx) => array[idx]);
}

function nms(boxes: number[][], threshold: number): number[] {
  const n = boxes.length;
  if (n === 0) return [];
  if (threshold < 0 || threshold > 1) {
    throw new Error('NMS threshold must be between 0 and 1');
  }

  const x1 = boxes.map((b) => b[0]);
  const y1 = boxes.map((b) => b[1]);
  const x2 = boxes.map((b) => b[2]);
  const y2 = boxes.map((b) => b[3]);

  const areas = x2.map((x2i, i) => (x2i - x1[i] + 1) * (y2[i] - y1[i] + 1));

  const keep: number[] = [];
  let order = Array.from({ length: n }, (_, i) => i);

  while (order.length > 0) {
    const i = order[0];
    keep.push(i);
    const orderTail = order.slice(1);

    if (orderTail.length === 0) break;

    const xx1 = orderTail.map((j) => Math.max(x1[i], x1[j]));
    const xx2 = orderTail.map((j) => Math.min(x2[i], x2[j]));
    const yy1 = orderTail.map((j) => Math.max(y1[i], y1[j]));
    const yy2 = orderTail.map((j) => Math.min(y2[i], y2[j]));

    const w = xx2.map((xx2j, idx) => Math.max(0, xx2j - xx1[idx] + 1));
    const h = yy2.map((yy2j, idx) => Math.max(0, yy2j - yy1[idx] + 1));
    const intersection = w.map((wj, idx) => wj * h[idx]);
    const union = orderTail.map((j) => areas[i] + areas[j] - intersection[orderTail.indexOf(j)]);

    order = intersection
      .map((inter, idx) => ({ inter, union: union[idx], origIdx: idx }))
      .filter(({ inter, union }) => inter / union <= threshold)
      .map(({ origIdx }) => orderTail[origIdx]);
  }

  return keep;
}

function reversedArgsort(array: number[]): number[] {
  const indices = Array.from({ length: array.length }, (_, i) => i);
  indices.sort((a, b) => array[b] - array[a]);
  return indices;
}

function preprocess(
  img: ImageData | HTMLImageElement | HTMLCanvasElement
): { scale: number; blob: Float32Array } {
  let imageData: ImageData;
  let imgWidth: number;
  let imgHeight: number;

  if (img instanceof ImageData) {
    imageData = img;
    imgWidth = img.width;
    imgHeight = img.height;
  } else if (img instanceof HTMLImageElement || img instanceof HTMLCanvasElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2d context');
    }
    canvas.width = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
    canvas.height = img instanceof HTMLImageElement ? img.naturalHeight : img.height;
    ctx.drawImage(img, 0, 0);
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    imgWidth = canvas.width;
    imgHeight = canvas.height;
  } else {
    throw new Error('Unsupported image type');
  }

  if (imgWidth === 0 || imgHeight === 0) {
    throw new Error('Image is empty');
  }

  const imgRatio = imgHeight / imgWidth;
  let nwidth: number;
  let nheight: number;

  if (imgRatio > IH / IW) {
    nheight = IH;
    nwidth = Math.max(1, Math.floor(nheight / imgRatio));
  } else {
    nwidth = IW;
    nheight = Math.max(1, Math.floor(nwidth * imgRatio));
  }

  const detScale = imgHeight / nheight;

  // Resize and pad image
  const canvas = document.createElement('canvas');
  canvas.width = IW;
  canvas.height = IH;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not get 2d context');
  }

  // Create temporary canvas for resized image
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = nwidth;
  tempCanvas.height = nheight;
  const tempCtx = tempCanvas.getContext('2d');
  if (!tempCtx) {
    throw new Error('Could not get 2d context');
  }

  // Draw original image to temp canvas (resized)
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = imgWidth;
  sourceCanvas.height = imgHeight;
  const sourceCtx = sourceCanvas.getContext('2d');
  if (!sourceCtx) {
    throw new Error('Could not get 2d context');
  }
  sourceCtx.putImageData(imageData, 0, 0);
  tempCtx.drawImage(sourceCanvas, 0, 0, nwidth, nheight);

  // Draw resized image to main canvas (padded)
  ctx.drawImage(tempCanvas, 0, 0);

  const finalImageData = ctx.getImageData(0, 0, IW, IH);

  // Convert to blob: shape [1, CH, IH, IW]
  const mean: [number, number, number] = [127.5, 127.5, 127.5];
  const std: [number, number, number] = [128.0, 128.0, 128.0];
  const blob = blobFromImage(finalImageData, mean, std);

  return { scale: detScale, blob };
}

function blobFromImage(
  img: ImageData,
  mean: [number, number, number],
  std: [number, number, number]
): Float32Array {
  if (img.width !== IW || img.height !== IH) {
    throw new Error(`Image must be ${IW}x${IH}, got ${img.width}x${img.height}`);
  }

  const blob = new Float32Array(1 * CH * IH * IW);
  const data = img.data;

  for (let c = 0; c < CH; c++) {
    for (let y = 0; y < IH; y++) {
      for (let x = 0; x < IW; x++) {
        const imgIdx = (y * IW + x) * 4; // RGBA
        const r = data[imgIdx];
        const g = data[imgIdx + 1];
        const b = data[imgIdx + 2];
        // Convert RGB to BGR and normalize
        const channels = [b, g, r]; // BGR order
        const pixelValue = channels[c];
        const idx = c * IH * IW + y * IW + x;
        blob[idx] = (pixelValue - mean[c]) / std[c];
      }
    }
  }

  return blob;
}

function postprocess(rawFaces: RawFaces, scale: number, threshold: Threshold): Face[] {
  const n = rawFaces.boxesShape[0];
  if (n === 0) return [];

  const scores2d = array1dTo2d(rawFaces.scores, n, 1);
  const boxes2d = array1dTo2d(rawFaces.boxes, n, 4);
  const keypoints2d = array1dTo2d(rawFaces.keypoints, n, 2 * KPS);

  // Scale boxes and keypoints
  const scaledBoxes = boxes2d.map((box) => box.map((v) => v * scale));
  const scaledKeypoints = keypoints2d.map((kp) => kp.map((v) => v * scale));

  // Sort by score (descending)
  const scores1d = scores2d.map((row) => row[0]);
  const order = reversedArgsort(scores1d);

  const sortedScores = take(scores2d, order);
  const sortedBoxes = take(scaledBoxes, order);
  const sortedKeypoints = take(scaledKeypoints, order);

  // Apply NMS
  const keep = nms(sortedBoxes, threshold.iou);

  const finalBoxes = take(sortedBoxes, keep);
  const finalScores = take(sortedScores, keep);
  const finalKeypoints = take(sortedKeypoints, keep);

  // Convert to Face objects
  const faces: Face[] = [];
  for (let i = 0; i < keep.length; i++) {
    const [x1, y1, x2, y2] = finalBoxes[i];
    const bbox = Bbox.fromXywh(x1, y1, x2 - x1, y2 - y1);

    const kps = finalKeypoints[i];
    const points: Point[] = [];
    for (let j = 0; j < KPS; j++) {
      points.push({ x: kps[j * 2], y: kps[j * 2 + 1] });
    }

    const keypoints: FaceKeypoints = {
      leftEye: points[0],
      rightEye: points[1],
      nose: points[2],
      leftMouth: points[3],
      rightMouth: points[4],
    };

    faces.push({
      probability: finalScores[i][0],
      keypoints,
      bbox,
    });
  }

  return faces;
}

// Helper functions for array manipulation
function array1dTo2d(arr: Float32Array | number[], rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result.push(Array.from(arr.slice(i * cols, (i + 1) * cols)));
  }
  return result;
}

function array2dTo1d(arr: number[][]): Float32Array {
  const flat = arr.flat();
  return new Float32Array(flat);
}

