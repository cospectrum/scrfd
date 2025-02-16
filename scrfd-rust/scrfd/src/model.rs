use crate::{Bbox, Face, FaceKeypoints, Point, Threshold};
use std::{io, path::Path};

use tract_ndarray::{prelude::*, Array, Array2, Array4, Data, RawData, RemoveAxis};
use tract_onnx::{
    prelude::{tract_itertools::Itertools, *},
    tract_core::anyhow::bail,
};

type Model = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug, Clone)]
pub struct Scrfd {
    model: Model,
    options: Options,
}

#[derive(Debug, Clone, Copy)]
struct Options {
    fmc: usize,
    num_anchors: usize,
    strides: [usize; 3],
}

impl Options {
    #[inline]
    const fn num_of_outputs(self) -> usize {
        self.strides.len() + 2 * self.fmc
    }
}

trait Take {
    type Taken;

    fn take(&self, indices: &[usize]) -> Self::Taken;
}

impl<A, S, D> Take for ArrayBase<S, D>
where
    A: Clone,
    S: Data + RawData<Elem = A>,
    D: Dimension + RemoveAxis,
{
    type Taken = Array<A, D>;

    fn take(&self, indices: &[usize]) -> Self::Taken {
        self.select(Axis(0), indices)
    }
}

impl Scrfd {
    pub fn from_model(model: Model) -> TractResult<Self> {
        let num_outputs = Self::num_of_outputs(&model)?;
        let options = match num_outputs {
            9 => Options {
                fmc: 3,
                num_anchors: 2,
                strides: [8, 16, 32],
            },
            _ => bail!("unsupported scrfd arch"),
        };
        assert_eq!(num_outputs, options.num_of_outputs());
        Ok(Self { model, options })
    }
    pub fn from_bytes(model_bytes: &[u8]) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut io::BufReader::new(model_bytes))?
            .into_optimized()?
            .into_runnable()?;
        Self::from_model(model)
    }
    pub fn from_path(path: impl AsRef<Path>) -> TractResult<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }
    pub fn detect(&self, img: &image::RgbImage) -> TractResult<Vec<Face>> {
        let threshold = Threshold {
            score: 0.4,
            iou: 0.5,
        };
        self.detect_with_threshold(img, threshold)
    }
    pub fn detect_with_threshold(
        &self,
        img: &image::RgbImage,
        threshold: Threshold,
    ) -> TractResult<Vec<Face>> {
        assert!(0.0 <= threshold.score && threshold.score <= 1.0);
        assert!(0.0 <= threshold.iou && threshold.iou <= 1.0);
        let (scale, blob) = preprocess(img)?;
        let raw_faces = self.forward(blob, threshold.score)?;
        Ok(self.postprocess(raw_faces, scale, threshold))
    }

    fn postprocess(
        &self,
        RawFaces {
            scores,
            boxes,
            keypoints,
        }: RawFaces,
        scale: f32,
        threshold: Threshold,
    ) -> Vec<Face> {
        let n = boxes.nrows();
        assert_eq!(scores.shape(), [n, 1]);
        assert_eq!(boxes.shape(), [n, 4]);
        assert_eq!(keypoints.shape(), [n, 2 * KPS]);

        let boxes = boxes.mapv_into(|el| el * scale);
        let keypoints = keypoints.mapv_into(|el| el * scale);

        let order = reversed_argsort(scores.as_slice().unwrap());

        let boxes = boxes.take(&order);
        let scores = scores.take(&order);
        let keep = nms(&boxes, scores.as_slice().unwrap(), threshold.iou);

        let boxes = boxes.take(&keep);
        let scores = scores.take(&keep);
        let keypoints = keypoints.take(&order).take(&keep);

        let mut faces = Vec::with_capacity(keep.len());
        for (i, &score) in scores.iter().enumerate() {
            let box_ = boxes.row(i);
            let kps = keypoints.row(i);
            assert_eq!(box_.shape(), [4]);
            assert_eq!(kps.shape(), [2 * KPS]);
            let (x1, y1, x2, y2) = (box_[0], box_[1], box_[2], box_[3]);
            let bbox = Bbox {
                x: x1,
                y: y1,
                w: x2 - x1,
                h: y2 - y1,
            };
            let kps = {
                let mut points = [Point { x: 0., y: 0. }; KPS];
                let it = kps.iter().copied();
                let xs = it.clone().step_by(2);
                let ys = it.skip(1).step_by(2);
                for ((x, y), point) in xs.zip_eq(ys).zip_eq(&mut points) {
                    *point = Point { x, y };
                }
                points
            };
            let kps = FaceKeypoints {
                left_eye: kps[0],
                right_eye: kps[1],
                nose: kps[2],
                left_mouth: kps[3],
                right_mouth: kps[4],
            };
            faces.push(Face {
                keypoints: kps,
                bbox,
                score,
            });
        }
        faces
    }

    fn forward(&self, blob: Array4<f32>, score_threshold: f32) -> TractResult<RawFaces> {
        assert_eq!(blob.shape(), [1, CH, IH, IW]);
        let outs = self.model.run(tvec!(Tensor::from(blob).into()))?;
        assert_eq!(outs.len(), self.options.num_of_outputs());

        let mut final_scores = Vec::new();
        let mut final_boxes = Vec::new();
        let mut final_keypoitns = Vec::new();

        for (idx, &stride) in self.options.strides.iter().enumerate() {
            assert_eq!(IW % stride, 0);
            assert_eq!(IH % stride, 0);
            let anchor_options = AnchorOptions {
                width: IW / stride,
                height: IH / stride,
                repeat: self.options.num_anchors,
                scale: stride as f32,
            };
            let anchor_centers = anchor_options.create();
            let n = anchor_centers.nrows();
            assert_eq!(anchor_centers.shape(), [n, 2]);

            let scores = outs[idx].to_array_view::<f32>()?;
            let boxes = outs[idx + self.options.fmc].to_array_view::<f32>()?;
            let keypoints = outs[idx + 2 * self.options.fmc].to_array_view::<f32>()?;
            assert!(scores.shape() == [n, 1] || scores.shape() == [1, n, 1]);
            assert!(boxes.shape() == [n, 4] || boxes.shape() == [1, n, 4]);
            assert!(keypoints.shape() == [n, 2 * KPS] || keypoints.shape() == [1, n, 2 * KPS]);
            let scores = scores.into_shape([n, 1])?;
            let boxes = boxes.into_shape([n, 4])?.mapv(|el| el * stride as f32);
            let keypoints = keypoints
                .into_shape([n, 2 * KPS])?
                .mapv(|el| el * stride as f32);
            let boxes = distance2boxes(&anchor_centers, &boxes);
            let keypoints = distance2kps(&anchor_centers, &keypoints);

            let likely: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s >= score_threshold)
                .map(|(i, _)| i)
                .collect();
            let scores = scores.take(&likely);
            let boxes = boxes.take(&likely);
            let keypoints = keypoints.take(&likely);
            final_scores.push(scores);
            final_boxes.push(boxes);
            final_keypoitns.push(keypoints);
        }
        let scores = final_scores.iter().map(|t| t.view()).collect::<Vec<_>>();
        let boxes = final_boxes.iter().map(|t| t.view()).collect::<Vec<_>>();
        let keypoints = final_keypoitns.iter().map(|t| t.view()).collect::<Vec<_>>();
        return Ok(RawFaces {
            scores: tract_ndarray::concatenate(Axis(0), &scores)?,
            boxes: tract_ndarray::concatenate(Axis(0), &boxes)?,
            keypoints: tract_ndarray::concatenate(Axis(0), &keypoints)?,
        });

        fn distance2boxes(points: &Array2<f32>, distance: &Array2<f32>) -> Array2<f32> {
            let n = points.nrows();
            assert_eq!(points.shape(), [n, 2]);
            assert_eq!(distance.shape(), [n, 4]);
            let x1 = points.column(0).to_owned() - distance.column(0);
            let y1 = points.column(1).to_owned() - distance.column(1);
            let x2 = points.column(0).to_owned() + distance.column(2);
            let y2 = points.column(1).to_owned() + distance.column(3);
            tract_ndarray::stack![Axis(1), x1, y1, x2, y2]
        }

        fn distance2kps(points: &Array2<f32>, distance: &Array2<f32>) -> Array2<f32> {
            let n = points.nrows();
            assert_eq!(points.shape(), [n, 2]);
            assert_eq!(distance.shape(), [n, 2 * KPS]);
            let mut kps = Vec::with_capacity(2 * KPS);
            for i in (0..2 * KPS).step_by(2) {
                let px = points.column(0).to_owned() + distance.column(i);
                let py = points.column(1).to_owned() + distance.column(i + 1);
                kps.push(px);
                kps.push(py);
            }
            debug_assert_eq!(kps.len(), 2 * KPS);
            let kps_view = kps.iter().map(|t| t.view()).collect::<Vec<_>>();
            tract_ndarray::stack(Axis(1), &kps_view).unwrap()
        }
    }
    fn num_of_outputs(model: &Model) -> TractResult<usize> {
        Ok(model.model().output_outlets()?.len())
    }
}

fn nms(boxes: &Array2<f32>, scores: &[f32], threshold: f32) -> Vec<usize> {
    let x1 = boxes.column(0);
    let y1 = boxes.column(1);
    let x2 = boxes.column(2);
    let y2 = boxes.column(3);

    let areas = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
    let mut keep = Vec::new();
    let mut order = reversed_argsort(scores);

    while let Some(&i) = order.first() {
        keep.push(i);
        let order_tail = &order[1..];

        let xx1 = x1.take(order_tail).mapv_into(|el| x1[i].max(el));
        let xx2 = x2.take(order_tail).mapv_into(|el| x2[i].min(el));
        let yy1 = y1.take(order_tail).mapv_into(|el| y1[i].max(el));
        let yy2 = y2.take(order_tail).mapv_into(|el| y2[i].min(el));

        let w = (xx2 - xx1 + 1.0).mapv_into(|el| el.max(0.0));
        let h = (yy2 - yy1 + 1.0).mapv_into(|el| el.max(0.0));
        let inter = w * h;
        let denum = areas[i] + areas.take(order_tail) - &inter;
        let ovr = inter / denum;

        order = ovr
            .iter()
            .enumerate()
            .filter(|&(_, &ov)| ov <= threshold)
            .map(|(idx, _)| order[idx + 1])
            .collect();
    }
    keep
}

fn reversed_argsort(array: &[f32]) -> Vec<usize> {
    let mut v = (0..array.len()).collect::<Vec<_>>();
    v.sort_by(|&a, &b| {
        array[b]
            .partial_cmp(&array[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

#[derive(Debug, Clone)]
struct RawFaces {
    scores: Array2<f32>,
    boxes: Array2<f32>,
    keypoints: Array2<f32>,
}

const KPS: usize = 5;

const IH: usize = 640;
const IW: usize = 640;
const CH: usize = 3;

fn preprocess(img: &image::RgbImage) -> TractResult<(f32, Array4<f32>)> {
    if img.is_empty() {
        bail!("image is empty")
    }
    let img_ratio = img.height() as f32 / img.width() as f32;
    let (nwidth, nheight) = if img_ratio > (IH as f32 / IW as f32) {
        let nheight = IH as u32;
        let nwidth = (nheight as f32 / img_ratio).floor().max(1.0) as _;
        (nwidth, nheight)
    } else {
        let nwidth = IW as u32;
        let nheight = (nwidth as f32 * img_ratio).floor().max(1.0) as _;
        (nwidth, nheight)
    };
    let det_scale = img.height() as f32 / nheight as f32;

    let img = {
        let mut bottom = image::RgbImage::new(IW as u32, IH as u32);
        let top =
            image::imageops::resize(img, nwidth, nheight, image::imageops::FilterType::Nearest);
        image::imageops::overlay(&mut bottom, &top, 0, 0);
        bottom
    };

    let mean = [127.5; 3];
    let std = [128.0; 3];
    Ok((det_scale, blob_from_image(&img, mean, std)))
}

fn blob_from_image(img: &image::RgbImage, mean: [f32; CH], std: [f32; CH]) -> Array4<f32> {
    assert_eq!(img.width(), IW as u32);
    assert_eq!(img.height(), IH as u32);
    Array4::from_shape_fn((1, CH, IH, IW), |(_, c, y, x)| {
        let mean = mean[c];
        let std = std[c];
        (img[(x as _, y as _)][c] as f32 - mean) / std
    })
}

#[derive(Clone, Copy)]
struct AnchorOptions {
    width: usize,
    height: usize,
    scale: f32,
    repeat: usize,
}

impl AnchorOptions {
    fn create(self) -> Array2<f32> {
        assert!(self.repeat > 0);
        assert!(self.width > 0);
        assert!(self.height > 0);

        let num_of_points = self.repeat * self.width * self.height;
        let mut buf = Vec::<f32>::with_capacity(num_of_points * 2);
        for y in 0..self.height {
            for x in 0..self.width {
                for _ in 0..self.repeat {
                    buf.push(self.scale * x as f32);
                    buf.push(self.scale * y as f32);
                }
            }
        }
        debug_assert_eq!(buf.len(), num_of_points * 2);
        Array2::from_shape_vec((num_of_points, 2), buf).unwrap()
    }
}

#[cfg(test)]
mod tests {}
