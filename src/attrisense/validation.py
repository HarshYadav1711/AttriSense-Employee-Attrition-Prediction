"""Shared validation types for inference and the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Single validation error or warning."""

    field: str
    message: str
    severity: str  # "error" | "warning"
