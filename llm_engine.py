from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai


load_dotenv()


@dataclass
class GeminiCodeGenerator:
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def __post_init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Put it in your .env file.")
        self.client = genai.Client(api_key=api_key)

    def generate_code(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")

        return self._strip_code_fences(text)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned