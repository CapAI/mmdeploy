# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_backend


@FUNCTION_REWRITER.register_rewriter("mmdet.models.roi_heads." "mask_heads.fcn_mask_head.FCNMaskHead.predict_by_feat")
def fcn_mask_head__predict_by_feat(
    self,
    mask_preds: Tuple[Tensor],
    results_list: List[Tensor],
    batch_img_metas: List[dict],
    rcnn_test_cfg: ConfigDict,
    rescale: bool = False,
    activate_map: bool = False,
) -> List[Tensor]:
    """Transform a batch of output features extracted from the head into mask
    results.

    Args:
        mask_preds (tuple[Tensor]): Tuple of predicted foreground masks,
            each has shape (n, num_classes, h, w).
        results_list (list[Tensor]): Detection results of
            each image.
        batch_img_metas (list[dict]): List of image information.
        rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        activate_map (book): Whether get results with augmentations test.
            If True, the `mask_preds` will not process with sigmoid.
            Defaults to False.

    Returns:
        list[Tensor]: Detection results of each image
        after the post process. Each item usually contains following keys.

            - dets (Tensor): Classification scores, has a shape
                (num_instance, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - masks (Tensor): Has a shape (num_instances, H, W).
    """
    ctx = FUNCTION_REWRITER.get_context()
    ori_shape = batch_img_metas[0]["img_shape"]
    dets, det_labels = results_list
    dets = dets.view(-1, 5)
    det_labels = det_labels.view(-1)
    backend = get_backend(ctx.cfg)
    mask_preds = mask_preds.sigmoid()
    bboxes = dets[:, :4]
    labels = det_labels
    threshold = rcnn_test_cfg.mask_thr_binary
    if not self.class_agnostic:
        box_inds = torch.arange(mask_preds.shape[0], device=bboxes.device)
        mask_pred = mask_preds[box_inds, labels][:, None]

    # grid sample is not supported by most engine
    # so we add a flag to disable it.
    mmdet_params = get_post_processing_params(ctx.cfg)
    export_postprocess_mask = mmdet_params.get("export_postprocess_mask", False)
    if not export_postprocess_mask:
        return mask_pred

    masks, _ = _do_paste_mask(mask_pred, bboxes, ori_shape[0], ori_shape[1], skip_empty=False)
    if backend == Backend.TENSORRT:
        return masks
    if threshold >= 0:
        masks = (masks >= threshold).to(dtype=torch.bool)
    return masks


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
