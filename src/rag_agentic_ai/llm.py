"""One place to get the course's chat model."""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def get_model(spec : str | None = None, **kwargs) -> BaseChatModel:
    """Return the model named by spec , or by the LLM_MODEL environment variable."""

    load_dotenv()
    model = spec or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("No model specified and LLM_MODEL environment variable not set.")
    return init_chat_model(model, **kwargs)