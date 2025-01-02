from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from onnxruntime import InferenceSession  # type: ignore
from PIL.Image import (
    Image as PILImage,
    Resampling,
)
from .schemas import Threshold


@dataclass
class Detections:
    bboxes: np.ndarray
    keypoints: np.ndarray

    @staticmethod
    def empty() -> Detections:
        return Detections(np.array([]), np.array([]))


@dataclass
class SCRFDBase:
    session: InferenceSession

    @classmethod
    def fmc(cls) -> int:
        return 3

    @classmethod
    def feat_stride_fpn(cls) -> list[int]:
        return [8, 16, 32]

    @classmethod
    def num_anchors(cls) -> int:
        return 2

    def output_names(self) -> list[str]:
        outputs = self.session.get_outputs()
        assert len(outputs) == 9, len(outputs)
        return [out.name for out in outputs]

    def input_name(self) -> str:
        inputs = self.session.get_inputs()
        assert len(inputs) == 1
        input = inputs[0]
        return input.name

    def forward(
        self, image: np.ndarray, scores_thresh: float
    ) -> tuple[list, list, list]:
        assert 0.0 <= scores_thresh <= 1.0
        assert image.ndim == 3
        input_height, input_width, _ = image.shape

        blob = self.blob_from_image(image, swap_rb=False)
        blob = np.expand_dims(blob, axis=0)
        net_outs = self.session.run(self.output_names(), {self.input_name(): blob})

        fmc = self.fmc()
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self.feat_stride_fpn()):
            scores = net_outs[idx][0]
            bbox_preds = net_outs[idx + fmc][0]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx + fmc * 2][0] * stride

            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1],  # type: ignore
                axis=-1,
            ).astype(np.float32)

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors() > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self.num_anchors(), axis=1
                ).reshape((-1, 2))

            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            pos_inds = np.where(scores >= scores_thresh)[0]
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            pos_kpss = kpss[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def resize(self, image: PILImage, *, width: int, height: int) -> PILImage:
        size = (width, height)
        return image.resize(size, resample=Resampling.NEAREST)

    def detect(self, image: PILImage, threshold: Threshold) -> Detections:
        N = 640
        assert image.mode == "RGB"

        img_ratio = image.height / image.width
        if img_ratio > 1.0:
            new_height = N
            new_width = int(new_height / img_ratio)
        else:
            new_width = N
            new_height = int(new_width * img_ratio)
        assert new_width <= N
        assert new_height <= N
        det_scale = image.height / new_height
        resized_img = self.resize(image, width=new_width, height=new_height)

        det_img = np.zeros((N, N, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = np.array(resized_img)

        scores_list, bboxes_list, kpss_list = self.forward(
            det_img, threshold.probability
        )
        if len(scores_list) == 0:
            return Detections.empty()

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) * det_scale
        kpss = np.vstack(kpss_list) * det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, threshold.nms)
        det = pre_det[keep, :]  # type: ignore

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]  # type: ignore
        return Detections(bboxes=det, keypoints=kpss)

    def nms(self, dets: np.ndarray, nms_thresh: float) -> list[int]:
        assert 0.0 <= nms_thresh <= 1.0
        assert dets.ndim == 2
        x1, y1, x2, y2, scores = dets.T

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # type: ignore
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def distance2bbox(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        N = len(points)
        assert points.shape == (N, 2)
        assert distance.shape == (N, 4)
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        N = len(points)
        assert points.shape == (N, 2)
        assert distance.shape == (N, 10)
        preds = []
        bound = distance.shape[1]
        for i in range(0, bound, 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def blob_from_image(
        self,
        img: np.ndarray,
        scaling: float = 1.0 / 128,
        mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
        swap_rb: bool = True,
    ) -> np.ndarray:
        assert img.ndim == 3
        assert img.shape[2] == len(mean)
        img = img.astype(dtype=np.float32, copy=True)
        img -= mean
        img *= scaling
        if swap_rb:
            img = img[..., ::-1]
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)
        return img
