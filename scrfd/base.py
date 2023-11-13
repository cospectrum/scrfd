from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from onnxruntime import InferenceSession
from PIL import Image
from PIL.Image import Image as PILImage

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
        input = self.session.get_inputs()[0]
        return input.name

    def forward(
        self,
        image: np.ndarray,
        scores_thresh: float,
    ) -> tuple[list, list, list]:
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_height = image.shape[0]
        input_width = image.shape[1]

        blob = self.blob_from_image(image, swap_rb=False)
        blob = np.expand_dims(blob, axis=0)
        net_outs = self.session.run(self.output_names(), {self.input_name(): blob})

        fmc = self.fmc()

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

            pos_inds = np.where(scores >= scores_thresh)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def resize(self, image: PILImage, *, width: int, height: int) -> PILImage:
        size = (width, height)
        return image.resize(size, resample=Image.NEAREST)

    def detect(
        self,
        image: PILImage,
        threshold: Threshold,
        max_num: int | None = None,
    ) -> Detections:
        im_ratio = image.height / image.width
        if im_ratio > 1.0:
            new_height = 640
            new_width = int(new_height / im_ratio)
        else:
            new_width = 640
            new_height = int(new_width * im_ratio)

        det_scale = new_height / image.height

        resized_img = self.resize(image, width=new_width, height=new_height)
        resized_img = np.array(resized_img)

        shape = (640, 640, 3)
        det_img = np.zeros(shape, dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(
            det_img, threshold.probability
        )
        if len(scores_list) == 0:
            return Detections.empty()

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, threshold.nms)
        det = pre_det[keep, :]  # type: ignore

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]  # type: ignore

        if max_num is not None and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])

            img_center = image.height // 2, image.width // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            kpss = kpss[bindex, :]

        return Detections(bboxes=det, keypoints=kpss)

    def nms(self, dets: np.ndarray, nms_thresh: float) -> list:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # type: ignore
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
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

    def distance2bbox(
        self,
        points: np.ndarray,
        distance: np.ndarray,
        max_shape: tuple | None = None,
    ) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])  # type: ignore
            y1 = y1.clamp(min=0, max=max_shape[0])  # type: ignore
            x2 = x2.clamp(min=0, max=max_shape[1])  # type: ignore
            y2 = y2.clamp(min=0, max=max_shape[0])  # type: ignore
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(
        self,
        points: np.ndarray,
        distance: np.ndarray,
        max_shape: tuple | None = None,
    ) -> np.ndarray:
        preds = []
        bound = distance.shape[1]
        for i in range(0, bound, 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])  # type: ignore
                py = py.clamp(min=0, max=max_shape[0])  # type: ignore
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
        img = img.astype(dtype=np.float32, copy=True)
        img -= mean
        img *= scaling
        if swap_rb:
            img = img[..., ::-1]
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 0, 2)
        return img
