from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    """配置类，包含应用程序的所有配置项"""
    api_key: str 
    api_url: str 
    model_name: str 

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量加载配置"""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        api_url = os.getenv("OPENAI_API_URL", "").strip()
        model_name = os.getenv("OPENAI_MODEL_NAME", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        if not api_url:
            raise ValueError("OPENAI_API_URL is not set in environment variables.")
        if not model_name:
            raise ValueError("OPENAI_MODEL_NAME is not set in environment variables.")
        return cls(api_key=api_key, api_url=api_url, model_name=model_name)