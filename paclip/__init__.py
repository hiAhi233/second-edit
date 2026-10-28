"""PA-CLIP2 public API for few-shot medical image anomaly detection.

Implementations live in models/data/prompts/engine/evaluation. Historical
text_side_anomaly imports forward to the same modules for checkpoint compatibility.
"""

from paclip.config import Config
from paclip.prompts import ThreeLevelPrompts, DEFAULT_BRAIN_MRI_PROMPTS, build_text_anchors
from paclip.models.text_adapter import ResidualTextAdapter
from paclip.models.text_inlayer import (
    InLayerBottleneckAdapter,
    OrganAdapterBank,
    install_inlayer_adapters,
)
from paclip.models.visual_inlayer import (
    install_visual_inlayer_adapters,
    uninstall_visual_inlayer_adapters,
    visual_encoder_blocks,
)
from paclip.models.paclip import TextSideAnomalyModel, load_trained_model
from paclip.losses import (
    text_separation_loss,
    global_alignment_loss,
    local_alignment_loss,
    TotalLoss,
)
from paclip.data.datasets import SliceAnomalyDataset, VolumeAnomalyDataset
from paclip.evaluation.metrics import image_metrics, pixel_metrics

__all__ = [
    "Config",
    "ThreeLevelPrompts",
    "DEFAULT_BRAIN_MRI_PROMPTS",
    "build_text_anchors",
    "ResidualTextAdapter",
    "InLayerBottleneckAdapter",
    "OrganAdapterBank",
    "install_inlayer_adapters",
    "install_visual_inlayer_adapters",
    "uninstall_visual_inlayer_adapters",
    "visual_encoder_blocks",
    "TextSideAnomalyModel",
    "load_trained_model",
    "text_separation_loss",
    "global_alignment_loss",
    "local_alignment_loss",
    "TotalLoss",
    "SliceAnomalyDataset",
    "VolumeAnomalyDataset",
    "image_metrics",
    "pixel_metrics",
]
