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
        return [out.name for out in outputs]

    def input_name(self) -> str:
        inputs = self.session.get_inputs()
        assert len(inputs) == 1
        return inputs[0].name

    def forward(
        self, image: np.ndarray, scores_thresh: float
    ) -> tuple[list, list, list]:
        CH, IH, IW = (3, 640, 640)
        FMC = self.fmc()

        assert 0.0 <= scores_thresh <= 1.0
        assert image.shape == (IH, IW, CH)
        ih, iw, _ = image.shape

        blob = self.blob_from_image(image, swap_rb=False)
        assert blob.shape == (CH, IH, IW)
        blob = np.expand_dims(blob, axis=0)
        assert blob.shape == (1, CH, IH, IW)

        output_names = self.output_names()
        net_outs: list[np.ndarray] = self.session.run(
            output_names, {self.input_name(): blob}
        )
        assert len(net_outs) == len(output_names)
        assert len(net_outs) == len(self.feat_stride_fpn()) + 2 * FMC

        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self.feat_stride_fpn()):
            anchor_height = ih // stride
            anchor_width = iw // stride
            anchor_centers = np.stack(
                np.mgrid[:anchor_height, :anchor_width][::-1],  # type: ignore
                axis=-1,
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors() > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self.num_anchors(), axis=1
                ).reshape((-1, 2))

            N = len(anchor_centers)
            assert anchor_centers.shape == (N, 2)

            scores = net_outs[idx]
            bbox_preds = net_outs[idx + FMC] * stride
            kps_preds = net_outs[idx + 2 * FMC] * stride

            assert scores.shape == (N, 1) or scores.shape == (1, N, 1)
            assert bbox_preds.shape == (N, 4) or bbox_preds.shape == (1, N, 4)
            assert kps_preds.shape == (N, 10) or kps_preds.shape == (1, N, 10)

            scores = scores.reshape((N, 1))
            bbox_preds = bbox_preds.reshape((N, 4))
            kps_preds = kps_preds.reshape((N, 10))

            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            indexes = np.where(scores >= scores_thresh)[0]
            assert indexes.shape == (len(indexes),)

            final_scores = scores[indexes]
            final_bboxes = bboxes[indexes]
            final_kpss = kpss[indexes]
            assert len(final_scores) == len(final_bboxes) == len(final_kpss)
            scores_list.append(final_scores)
            bboxes_list.append(final_bboxes)
            kpss_list.append(final_kpss)

        return scores_list, bboxes_list, kpss_list

    @classmethod
    def resize(cls, image: PILImage, *, width: int, height: int) -> PILImage:
        assert height > 0, height
        assert width > 0, width
        size = (width, height)
        return image.resize(size, resample=Resampling.NEAREST)

    def detect(self, image: PILImage, threshold: Threshold) -> Detections:
        (IH, IW, CH) = (640, 640, 3)
        assert image.mode == "RGB"

        img_ratio = image.height / image.width
        if img_ratio > 1.0:
            new_height = IH
            new_width = max(1, int(new_height / img_ratio))
        else:
            new_width = IW
            new_height = max(1, int(new_width * img_ratio))
        assert 1 <= new_width <= IW, new_width
        assert 1 <= new_height <= IH, new_height
        det_scale = image.height / new_height
        resized_img = self.resize(image, width=new_width, height=new_height)

        det_img = np.zeros((IH, IW, CH), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = np.array(resized_img)

        scores_list, bboxes_list, kpss_list = self.forward(
            det_img, threshold.probability
        )
        assert len(scores_list) == len(bboxes_list) == len(kpss_list)
        if len(scores_list) == 0:
            return Detections.empty()

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) * det_scale
        kpss = np.vstack(kpss_list) * det_scale

        dets = np.hstack((bboxes, scores)).astype(np.float32)
        dets = dets[order, :]
        keep = self.nms(dets, threshold.nms)
        final_dets = dets[keep, :]
        final_kpss = kpss[order, ...][keep, ...]
        return Detections(bboxes=final_dets, keypoints=final_kpss)

    @classmethod
    def nms(cls, dets: np.ndarray, nms_thresh: float) -> list[int]:
        assert 0.0 <= nms_thresh <= 1.0
        assert dets.ndim == 2
        x1, y1, x2, y2, scores = dets.T

        areas: np.ndarray = (x2 - x1 + 1) * (y2 - y1 + 1)
        order: np.ndarray = scores.argsort()[::-1]

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
        assert distance.shape == (N, 4), distance.shape
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        N = len(points)
        assert points.shape == (N, 2)
        assert distance.shape == (N, 10), distance.shape
        preds = []
        bound = distance.shape[1]
        for i in range(0, bound, 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    @classmethod
    def blob_from_image(
        cls,
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
