"""Prompts for the optional PneumoniaMNIST benchmark."""
from .base import ThreeLevelPrompts

PNEUMONIA_PROMPTS = ThreeLevelPrompts(
    normal={
        "1": ["a normal chest x-ray"],
        "2": ["a normal chest x-ray with clear lungs"],
        "3": ["a normal chest x-ray with sharp lung markings"],
    },
    abnormal={
        "1": ["a chest x-ray with pneumonia"],
        "2": ["a chest x-ray with pulmonary opacities"],
        "3": ["a chest x-ray with hazy infiltrates"],
    },
)
