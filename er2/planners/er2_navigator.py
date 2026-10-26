"""Gemini Robotics ER 2 planner for llm_mjx_go1_playground.

This is a NEW file. It does NOT modify the original
src/planning/llm_nagivator.py (GeminiThinkingNavigator), which is kept
intact so the original planner can still be used.

Difference from the original planner (important):
- Original: keeps a chat session (client.chats.create) and calls
  chat.send_message([prompt]) with model
  "gemini-2.0-flash-thinking-exp".
- This adapter: makes one stateless call per plan() via
  client.interactions.create(model="gemini-robotics-er-2-preview"),
  following the official Gemini Robotics ER 2 Python example
  (interactions.create, input=[{"type": "text", "text": ...}],
  generation_config={"thinking_level": "high"}, result in output_text).

Statelessness is safe for this repo's trial-and-error loop because
src/mission_executer.py already puts the feedback into the prompt
itself: after a failed attempt it builds the next prompt from
_format_failure_message(result) and passes it to plan(prompt=...).
So no chat history is needed for the retry to learn from failure.

Output contract (same as the original): plan() returns an
LLMNavigationPlan whose waypoints are a list of np.ndarray (x, y).
The model is instructed to return ONLY a Python list of (x, y)
tuples, so the existing _clean_instruction + eval parsing can be
reused unchanged. Vision input (image=...) is not used by this
text-only adapter; extra kwargs are accepted and ignored so
mission_executer-style callers do not break.

Requires a Gemini API key with access to
gemini-robotics-er-2-preview, supplied via the genai.Client
(typically GOOGLE_API_KEY / GEMINI_API_KEY env vars). No key is
stored in this file, and no API call is made at import time.
"""

from typing import Any, List

import numpy as np
from google import genai

from src.planning.base import Planner
from src.planning.llm_nagivator import (
    LLMNavigationPlan,
    _clean_instruction,
    _convert_to_list_of_numpy_arrays,
)

ER2_MODEL = "gemini-robotics-er-2-preview"

_FORMAT_SUFFIX = (
    "\nReturn ONLY a Python list of (x, y) waypoint tuples, "
    "for example [(1.0, 2.0), (3.0, 4.0)], with no other text, "
    "no markdown, and no explanation."
)


class GeminiRoboticsER2Navigator(Planner):
    """Text-only navigation planner backed by Gemini Robotics ER 2."""

    def __init__(
        self,
        model: genai.Client,
        model_name: str = ER2_MODEL,
    ):
        self.model = model
        self.model_name = model_name

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
                "Gemini Robotics ER 2 returned no text output; "
                "check the API key, model access, and response."
            )
        waypoints = eval(_clean_instruction(text))  # Same parsing as original
        waypoints = _convert_to_list_of_numpy_arrays(waypoints)
        return LLMNavigationPlan(
            waypoints=waypoints, trajectory=[], prompt=prompt
        )

    def reset_chat(self) -> None:
        """No-op: ER 2 interactions are stateless, there is no chat to reset.

        Kept for compatibility with callers that reset the original
        GeminiThinkingNavigator between missions.
        """
        return None
