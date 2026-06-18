from typing import Literal, Optional, Union, overload

import openai
from google import genai

from .config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    Provider,
)

_openai_client: Optional[openai.OpenAI] = None
_gemini_client: Optional[genai.Client] = None


@overload
def client(provider: Literal["openai"]) -> openai.OpenAI: ...
@overload
def client(provider: Literal["gemini"]) -> genai.Client: ...


def client(provider: Provider) -> Union[openai.OpenAI, genai.Client]:
    global _openai_client, _gemini_client
    if provider == "openai":
        if _openai_client is None:
            _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return _openai_client
    if provider == "gemini":
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        return _gemini_client
    raise ValueError(f"Unknown provider: {provider}")
