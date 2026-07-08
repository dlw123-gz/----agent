from __future__ import annotations

import base64
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(slots=True)
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "OpenAICompatibleClient":
        load_env_file(env_file)
        return cls(
            base_url=os.environ["LLM_BASE_URL"].rstrip("/"),
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1200")),
        }
        if "json" in f"{system_prompt}\n{user_prompt}".lower():
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"LLM request timed out after {self.timeout_seconds}s. "
                "Increase LLM_TIMEOUT_SECONDS in .env, check network/proxy access to the API host, "
                "or use a faster/smaller model."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"LLM request timed out after {self.timeout_seconds}s. "
                "Increase LLM_TIMEOUT_SECONDS in .env, check network/proxy access to the API host, "
                "or use a faster/smaller model."
            ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        return _extract_message_content(data)

    def complete_with_image(self, prompt: str, image_path: str | Path) -> str:
        image_data = _image_data_url(image_path)
        payload = {
            "model": os.environ.get("LLM_VISION_MODEL", self.model),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1200")),
        }
        if "json" in prompt.lower():
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"LLM vision request timed out after {self.timeout_seconds}s. "
                "Increase LLM_TIMEOUT_SECONDS in .env or use a faster vision-capable model."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"LLM vision request timed out after {self.timeout_seconds}s. "
                "Increase LLM_TIMEOUT_SECONDS in .env or use a faster vision-capable model."
            ) from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if "image_url" in error_body or "vision" in error_body.lower():
                raise RuntimeError(
                    "Configured LLM provider/model does not support image input. "
                    "Vision shop recognition cannot be exact with this provider; "
                    "set LLM_VISION_MODEL/LLM_BASE_URL to a vision-capable OpenAI-compatible model "
                    "or turn off Vision shop exact."
                ) from exc
            raise RuntimeError(f"LLM vision HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM vision connection failed: {exc.reason}") from exc
        return _extract_message_content(data)


def load_env_file(env_file: str | Path | None = None) -> None:
    path = Path(env_file) if env_file else _default_env_path()
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_env_path() -> Path:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    return Path(__file__).resolve().parents[2] / ".env"


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_message_content(data: dict) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM response missing choices[0].message: {json.dumps(data, ensure_ascii=False)[:1000]}") from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise RuntimeError(f"LLM returned an empty message: {json.dumps(data, ensure_ascii=False)[:1000]}")
