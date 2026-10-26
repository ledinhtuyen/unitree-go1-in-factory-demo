"""Gemini Robotics ER 2 planner via Vertex AI for llm_mjx_go1_playground.

This is a NEW file. It does NOT modify the original
src/planning/llm_nagivator.py, nor src/planning/er2_navigator.py
(the Gemini Developer API version).

Difference from er2_navigator.py:
- That file takes a genai.Client built for the Gemini Developer API
  (API key, generativelanguage.googleapis.com).
- This file builds a Vertex AI client instead:
      genai.Client(vertexai=True, project=..., location=...)
  or, when no client is passed, from the environment variables
  GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (default location
  'us-central1' when the variable is unset).

Billing therefore goes through the GCP project, not through AI Studio
prepayment credits. Authentication on a machine outside GCP needs a
GCP credential, for example a service account whose key file is
pointed to by GOOGLE_APPLICATION_CREDENTIALS, or application default
credentials. No credential value is stored in this file.

Whether the interactions API for gemini-robotics-er-2-preview is
available on Vertex AI, and in which location, was NOT verified when
this file was written. A test call is what finds out: a model-not-found
or not-available-in-location error means ER 2 interactions are not
on Vertex AI in that project/location, even if ordinary Gemini
generateContent models work there.

Output contract is the same as the other planners: plan() returns an
LLMNavigationPlan whose waypoints are a list of np.ndarray (x, y).
The model is instructed to return ONLY a Python list of (x, y)
tuples, so the existing _clean_instruction + eval parsing is reused.
This is a text-only adapter; the vision variant (image=...) is not
covered. The planner is stateless per plan() call, which is safe for
this repo because mission_executer.py already puts trial-and-error
feedback into the next prompt itself.
"""

import os
from typing import Any, Optional

import numpy as np
from google import genai

from src.planning.base import Planner
from src.planning.llm_nagivator import (
    LLMNavigationPlan,
    _clean_instruction,
    _convert_to_list_of_numpy_arrays,
)

ER2_MODEL = "gemini-robotics-er-2-preview"
DEFAULT_LOCATION = "us-central1"

_FORMAT_SUFFIX = (
    "\nReturn ONLY a Python list of (x, y) waypoint tuples, "
    "for example [(1.0, 2.0), (3.0, 4.0)], with no other text, "
    "no markdown, and no explanation."
)


class GeminiRoboticsER2VertexNavigator(Planner):
    """Text-only ER 2 navigation planner backed by Vertex AI."""

    def __init__(
        self,
        model: Optional[genai.Client] = None,
        model_name: str = ER2_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.model_name = model_name
        if model is not None:
            self.model = model
            return
        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        resolved_location = (
            location or os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION
        )
        if not resolved_project:
            raise ValueError(
                "Vertex AI needs a GCP project: pass project=... or set "
                "GOOGLE_CLOUD_PROJECT. No credential value is accepted here."
            )
        self.project = resolved_project
        self.location = resolved_location
        self.model = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=resolved_location,
        )

    def plan(self, prompt: str, **kwargs: Any) -> LLMNavigationPlan:
        response = self.model.interactions.create(
            model=self.model_name,
            input=[{"type": "text", "text": prompt + _FORMAT_SUFFIX}],
            generation_config={"thinking_level": "high"},
        )
        text = getattr(response, "output_text", None) or getattr(
            response, "text", ""
        )
        if not text:
            raise ValueError(
                "Gemini Robotics ER 2 on Vertex AI returned no text output; "
                "check the GCP project, location, IAM permission, and response."
            )
        waypoints = eval(_clean_instruction(text))  # Same parsing as original
        waypoints = _convert_to_list_of_numpy_arrays(waypoints)
        return LLMNavigationPlan(
            waypoints=waypoints, trajectory=[], prompt=prompt
        )

    def reset_chat(self) -> None:
        """No-op: ER 2 interactions are stateless, there is no chat to reset."""
        return None
