import type { Face } from './scrfd/types';

export function drawFaces(
  ctx: CanvasRenderingContext2D,
  faces: Face[],
  options: { boxColor?: string; pointRadius?: number } = {}
) {
  const { boxColor = '#ff0000', pointRadius = 4 } = options;

  for (const face of faces) {
    const { bbox, keypoints } = face;

    // Draw keypoints (circles)
    const points = [
      keypoints.leftEye,
      keypoints.rightEye,
      keypoints.nose,
      keypoints.leftMouth,
      keypoints.rightMouth,
    ];
    ctx.fillStyle = boxColor;
    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, pointRadius, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw bounding box
    const { upperLeft, lowerRight } = bbox;
    const x = upperLeft.x;
    const y = upperLeft.y;
    const width = bbox.width();
    const height = bbox.height();

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
  }
}
