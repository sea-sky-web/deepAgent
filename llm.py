from __future__ import annotations

import json
from typing import Type

from openai import OpenAI
from pydantic import BaseModel

from config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.api_url,
        )

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel],
    ) -> BaseModel:
        schema_json = json.dumps(
            output_model.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
你必须输出严格 JSON，且必须符合下面的 JSON Schema。
不要输出 markdown，不要输出解释，不要输出代码块，只输出 JSON。

JSON Schema:
{schema_json}

用户任务:
{user_prompt}
""".strip()

        text = self.chat_text(system_prompt=system_prompt, user_prompt=prompt)

        try:
            return output_model.model_validate_json(text)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse model output as {output_model.__name__}. Raw output: {text}"
            ) from exc