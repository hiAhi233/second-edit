"""Task prompt registry, independent of training and visualization commands."""
from .base import ThreeLevelPrompts, build_text_anchors, resolve_prompt_set
from .brain import (
    LEGACY_BRAIN_MRI_PROMPTS, BRAIN_MRI_SENTENCE_PROMPTS, BRAIN_MRI_PROMPT_SETS,
    DEFAULT_BRAIN_MRI_PROMPT_SET, DEFAULT_BRAIN_MRI_PROMPTS,
)
from .thymoma import THYMOMA_SENTENCE_PROMPTS

TASK_PROMPT_SETS = {
    "brain_mri": BRAIN_MRI_PROMPT_SETS,
    "thymoma": {"thymoma_sentence": THYMOMA_SENTENCE_PROMPTS},
}
DEFAULT_TASK_PROMPT_SETS = {"brain_mri": DEFAULT_BRAIN_MRI_PROMPT_SET, "thymoma": "thymoma_sentence"}
