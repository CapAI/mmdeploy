from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

import torch
from mmengine.dataset.utils import default_collate
from mmengine import Config
from mmengine.registry import Registry
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmdeploy.utils import Codebase, Task, get_backend, get_codebase_config, load_config
from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.codebase.mmrotate.deploy.rotated_detection import RotatedDetection
from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import __BACKEND_MODEL  # type: ignore

from cap_mmlab.utils import register_all_modules
# from cap_mmlab.core.collate import video_detection_collate


CAPMMLAB_TASK_REGISTRY = Registry("cap_mmlab_tasks")


@CODEBASE.register_module(Codebase.CAPMMLAB.value)
class CapMMLab(MMCodebase):
    task_registry = CAPMMLAB_TASK_REGISTRY

    @classmethod
    def register_deploy_modules(cls):
        from mmdeploy.codebase.mmdet.deploy.object_detection import MMDetection
        from mmdeploy.codebase.mmrotate.deploy.rotated_detection import MMRotate

        MMRotate.register_deploy_modules()
        MMDetection.register_deploy_modules()

    @classmethod
    def register_all_modules(cls):
        from cap_mmlab.utils import register_all_modules

        cls.register_deploy_modules()
        register_all_modules()


def process_model_config(
    model_cfg: Config, imgs: Sequence[str] | Sequence[np.ndarray], input_shape: Sequence[int] | None = None
):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """
    cfg = model_cfg.copy()

    pipeline = cfg.test_pipeline
    pipeline = [transform for transform in pipeline if "LoadAnnotations" not in transform.type]
    cfg.test_pipeline = pipeline
    return cfg


def build_detection_model(
    model_files: Sequence[str],
    deploy_cfg: str | Config,
    device: str,
    data_preprocessor: Config | BaseDataPreprocessor | None = None,
    **kwargs,
):
    """Build rotated detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Rotated detector for a configured backend.
    """
    # load cfg if necessary
    (deploy_cfg,) = load_config(deploy_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get("model_type", "end2end")

    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs,
        )
    )
    return backend_detector

def create_video_infos(imgs: list[str]) -> dict:
    """ mock video infos from a small inference set"""
    video_infos = {
        'images': [
            {
                'img_id': Path(img).name,
                'file_name': img,
                'img_path': img,
                'lon': 4.009455682918974,
                'lat': 51.95734364569949,
                'theta': 5.2,
                'radar_range': 1533,
                'instances': []
            } for img in imgs],
        'video_length': len(imgs),
        'video_id': 0,
    }
    return video_infos

@CAPMMLAB_TASK_REGISTRY.register_module(Task.ROTATED_VIDEO_DETECTION.value)
class RotatedVideoDetection(RotatedDetection):
    def __init__(self, *args, **kwargs):
        register_all_modules()
        super().__init__(*args, **kwargs)

    def build_backend_model(self, model_files: Optional[str] = None, **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """

        model = build_detection_model(model_files, self.deploy_cfg, device=self.device, data_preprocessor=None)
        model = model.to(self.device)
        return model.eval()

    def build_pytorch_model(
        self, model_checkpoint: Optional[str] = None, cfg_options: Optional[dict] = None, **kwargs
    ) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.
            cfg_options (dict): Optional config key-pair parameters.

        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import MODELS

        model = deepcopy(self.model_cfg.model)
        model.pop("pretrained", None)
        model = MODELS.build(model)
        if model_checkpoint is not None:
            from mmengine.runner.checkpoint import load_checkpoint

            load_checkpoint(model, model_checkpoint, map_location=self.device)

        model = revert_sync_batchnorm(model)
        if hasattr(model, "backbone") and hasattr(model.backbone, "switch_to_deploy"):
            model.backbone.switch_to_deploy()
        model = model.to(self.device)
        model.eval()
        return model

    def create_input(
        self,
        imgs: str | np.ndarray,
        input_shape: Sequence[int] | None = None,
        data_preprocessor: BaseDataPreprocessor | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Create input for rotated object detection.

        Args:
            imgs (str | np.ndarray): Input image(s), accepted data type are
            `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmengine.dataset import Compose

        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError("imgs must be strings or numpy arrays")
        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError("imgs must be strings or numpy arrays")
        cfg = process_model_config(self.model_cfg, imgs, input_shape)

        pipeline = cfg.test_pipeline
        test_pipeline = Compose(pipeline)

        if (
        any(['RefFrameSample' in x['type'] for x in pipeline] ) or
        any(['TransformBroadcaster' in x['type'] for x in pipeline])
        ):
            video_infos = create_video_infos(imgs)
            data = test_pipeline(video_infos)

        else:
            raise NotImplementedError("Only support video pipeline through transform broadcaster for now")

        data = default_collate([data])
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data["inputs"]
        else:
            return data, BaseTask.get_tensor_from_input(data)
