"""Model configuration for the ladder.

Persists the user's choice of mode ('default' | 'local' | 'api_key'),
model name, and optional base URL to ~/.llm-ladder/model_config.json.

Note: API keys are NEVER stored here. The key is handled entirely
elsewhere (see the LLM_LADDER_API_KEY environment variable consumers).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

CONFIG_PATH = os.path.expanduser("~/.llm-ladder/model_config.json")

VALID_MODES = {"default", "local", "api_key"}


@dataclass
class ModelConfig:
    mode: str
    model: str
    base_url: str | None = None
    api_style: str = "openai"


def _validate(cfg: ModelConfig) -> bool:
    return cfg.mode in VALID_MODES


def load_config() -> ModelConfig:
    """Load the model config from disk.

    Returns the default config (mode='default', model='', base_url=None)
    if the file does not exist or contains invalid JSON.
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ModelConfig(mode="default", model="", base_url=None)

    mode = data.get("mode")
    if mode not in VALID_MODES:
        return ModelConfig(mode="default", model="", base_url=None)

    api_style = data.get("api_style")
    if api_style not in ("openai", "anthropic"):
        api_style = "openai"

    return ModelConfig(
        mode=mode,
        model=str(data.get("model", "")),
        base_url=data.get("base_url"),
        api_style=api_style,
    )


def save_config(cfg: ModelConfig) -> None:
    """Write the model config to disk, creating the directory if needed.

    The payload never includes an API key — this module has no notion of
    one by design.
    """
    if not _validate(cfg):
        raise ValueError(
            f"invalid mode {cfg.mode!r}; expected one of {sorted(VALID_MODES)}"
        )

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    payload = {
        "mode": cfg.mode,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_style": cfg.api_style,
    }
    # Write atomically: dump to a temp file in the same directory, then
    # rename onto CONFIG_PATH so a concurrent load_config() never sees a
    # partially-written file. The temp name is unique per write (pid +
    # random suffix) so concurrent saves under ThreadingHTTPServer can't
    # race on the same temp path.
    tmp_path = f"{CONFIG_PATH}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, CONFIG_PATH)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
