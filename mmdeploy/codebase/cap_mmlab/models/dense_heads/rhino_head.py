from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter("cap_mmlab.models.dense_heads.RHINOHead.predict_by_feat")
@FUNCTION_REWRITER.register_rewriter("cap_mmlab.models.dense_heads.RHINOPositiveHungarianHead.predict_by_feat")
@FUNCTION_REWRITER.register_rewriter(
    "cap_mmlab.models.dense_heads.RHINOPositiveHungarianClassificationHead.predict_by_feat"
)
def detrhead__predict_by_feat__default(
    self,
    all_cls_scores_list: List[Tensor],
    all_bbox_preds_list: List[Tensor],
    batch_img_metas: List[dict],
    rescale: bool = True,
):
    """Rewrite `predict_by_feat` of `FoveaHead` for default backend."""

    cls_scores = all_cls_scores_list[-1]
    bbox_preds = all_bbox_preds_list[-1]

    img_shape = batch_img_metas[0]["img_shape"]
    max_per_img = self.test_cfg.get("max_per_img", len(cls_scores[0]))
    batch_size = cls_scores.size(0)
    # `batch_index_offset` is used for the gather of concatenated tensor
    if self.loss_cls.use_sigmoid:
        batch_index_offset = torch.arange(batch_size).to(cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(batch_size, max_per_img)
        scores, indexes = cls_scores.sigmoid().flatten(1).topk(max_per_img, dim=1)
        det_labels = indexes % self.num_classes
        bbox_index = torch.div(indexes, self.num_classes, rounding_mode="trunc")
        bbox_preds = bbox_preds.view(-1, 5)[bbox_index]
        bbox_preds = bbox_preds.view(batch_size, -1, 5)
    else:
        scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(max_per_img)
        batch_inds = torch.arange(batch_size, device=scores.device).unsqueeze(-1)
        bbox_preds = bbox_preds[batch_inds, bbox_index, ...]
        # add unsqueeze to support tensorrt
        det_labels = det_labels.unsqueeze(-1)[batch_inds, bbox_index, ...].squeeze(-1)
    det_bboxes = bbox_preds

    h = torch.Tensor([img_shape[-2]]).to(det_bboxes.device)
    w = torch.Tensor([img_shape[-1]]).to(det_bboxes.device)

    det_bboxes[:, :, 0:4:2] = det_bboxes[:, :, 0:4:2].clamp_(min=0, max=1) * h
    det_bboxes[:, :, 1:4:2] = det_bboxes[:, :, 1:4:2].clamp_(min=0, max=1) * w

    # denormalize the angle dimension
    det_bboxes[:, :, 4] = det_bboxes[:, :, 4] * self.angle_factor
    det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)
    return det_bboxes, det_labels
