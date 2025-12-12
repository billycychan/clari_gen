"""Clarification generation prompts with different variants."""

from .at_standard import ClarificationATStandardPrompt
from .at_cot import ClarificationATCoTPrompt

__all__ = [
    "ClarificationATStandardPrompt",
    "ClarificationATCoTPrompt",
]
