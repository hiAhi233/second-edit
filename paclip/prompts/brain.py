"""Versioned brain MRI prompt sets. Historical names retain their wording."""
from .base import ThreeLevelPrompts

LEGACY_BRAIN_MRI_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal brain MRI"],
        "2": ["a normal brain MRI with intact anatomy and clear ventricles"],
        "3": ["a normal brain MRI with homogeneous tissue texture and sharp boundaries"],
    },
    abnormal={
        "1": ["an abnormal brain MRI"],
        "2": ["an abnormal brain MRI with a focal lesion"],
        "3": ["an abnormal brain MRI with irregular texture and blurred boundaries"],
    },
)

BRAIN_MRI_SENTENCE_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal brain MRI"],
        "2": ["brain MRI showing preserved brain anatomy and symmetric ventricles"],
        "3": ["preserved gray-white matter contrast with well-defined tissue boundaries on brain MRI"],
    },
    abnormal={
        "1": ["an abnormal brain MRI"],
        "2": ["brain MRI showing a focal lesion in the brain parenchyma"],
        "3": ["heterogeneous lesion signal with irregular indistinct margins on brain MRI"],
    },
)

BRAIN_MRI_PROMPT_SETS = {
    "brain_mri": LEGACY_BRAIN_MRI_PROMPTS,
    "brain_mri_sentence": BRAIN_MRI_SENTENCE_PROMPTS,
}

DEFAULT_BRAIN_MRI_PROMPT_SET = "brain_mri_sentence"

DEFAULT_BRAIN_MRI_PROMPTS = BRAIN_MRI_PROMPT_SETS[DEFAULT_BRAIN_MRI_PROMPT_SET]
