"""Recorded, stateless calls to a local Ollama server using its native API."""

import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from benchmark.contracts import LLMCallError, LLMResponse, Message, ModelSettings, Usage

_OPTIONS = {
    "num_ctx",
    "num_predict",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "num_thread",
}


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        file.write(text + "\n")


def _object(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise LLMCallError("Ollama returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMCallError("Ollama response must be a JSON object")
    if value.get("error"):
        raise LLMCallError(f"Ollama error: {value['error']}")
    return value


def _usage(raw: dict) -> Usage:
    counts = []
    for key in ("prompt_eval_count", "eval_count"):
        value = raw.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise LLMCallError(f"Invalid Ollama token count: {key}")
        counts.append(value)
    return Usage(*counts)


class OllamaClient:
    """No retained history, implicit retries, model downloads, or context truncation.

    Every call reserves a new directory before any HTTP request. Filesystem errors
    propagate unchanged: an incomplete evidence record must stop the caller.
    """

    def __init__(
        self,
        artifact_root: Path,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600,
        expected_digest: str | None = None,
    ) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("Ollama base URL must be a local HTTP origin without credentials")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.artifact_root = Path(artifact_root)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.expected_digest = expected_digest
        # Local requests must not go through an ambient HTTP/SOCKS proxy.
        self._opener = build_opener(ProxyHandler({}))

    def _http(self, endpoint: str, payload: dict | None = None) -> bytes:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(
            self.base_url + endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMCallError(f"Ollama HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError, ConnectionError, OSError, HTTPException) as exc:
            raise LLMCallError(f"Ollama transport error: {exc}") from exc

    def _model(self, name: str) -> dict:
        models = _object(self._http("/api/tags")).get("models", [])
        if not isinstance(models, list):
            raise LLMCallError("Invalid Ollama model list")
        selected = next(
            (model for model in models if isinstance(model, dict) and model.get("name") == name),
            None,
        )
        if selected is None:
            raise LLMCallError(f"Model {name!r} is not installed locally; run ollama pull first")
        if not isinstance(selected.get("digest"), str) or not selected["digest"]:
            raise LLMCallError("Local model has no digest")
        if selected.get("remote_host") or selected.get("remote_model"):
            raise LLMCallError("Cloud models are not allowed in a local experiment")
        return selected

    @staticmethod
    def _payload(messages: tuple[Message, ...], settings: ModelSettings) -> dict:
        if settings.provider != "ollama" or not settings.model or ":" not in settings.model:
            raise ValueError("Use provider='ollama' and an explicit model tag")
        if settings.model.endswith(":cloud"):
            raise ValueError("Cloud models are not allowed")
        if not messages or any(
            m.role not in {"system", "user", "assistant"} or not isinstance(m.content, str)
            for m in messages
        ):
            raise ValueError("Provide nonempty messages with supported roles and text content")
        params = dict(settings.parameters)
        unknown = params.keys() - {"options", "think", "keep_alive"}
        if unknown:
            raise ValueError(f"Unsupported Ollama parameters: {sorted(unknown)}")
        options = params.get("options", {})
        if not isinstance(options, dict) or options.keys() - _OPTIONS:
            raise ValueError("Unsupported Ollama generation options")
        for key in ("num_ctx", "num_predict"):
            if type(options.get(key)) is not int or options[key] <= 0:
                raise ValueError(f"Set an explicit positive {key}")
        if type(params.get("think")) is not bool:
            raise ValueError("Set think explicitly to true or false")
        payload = {
            "model": settings.model,
            "messages": [asdict(message) for message in messages],
            **params,
            "stream": False,
            "truncate": False,
            "shift": False,
        }
        # Snapshot mutable settings so later calls cannot change this request.
        return json.loads(json.dumps(payload, allow_nan=False))

    def generate(self, messages: tuple[Message, ...], settings: ModelSettings) -> LLMResponse:
        payload = self._payload(messages, settings)
        call_id = uuid4().hex
        directory = self.artifact_root / "calls" / call_id
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "request.json", payload)
        started = time.monotonic()
        context = {
            "call_id": call_id,
            "provider": "ollama",
            "base_url": self.base_url,
            "started_at": datetime.now(UTC).isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "expected_digest": self.expected_digest,
        }
        _write_json(directory / "client.json", context)
        raw = None
        try:
            model = self._model(payload["model"])
            if self.expected_digest is not None and model["digest"] != self.expected_digest:
                raise LLMCallError("Installed model digest differs from expected_digest")
            version = _object(self._http("/api/version"))
            details = _object(self._http("/api/show", {"model": payload["model"]}))
            if details.get("remote_host") or details.get("remote_model"):
                raise LLMCallError("Cloud models are not allowed in a local experiment")
            _write_json(
                directory / "environment.json",
                {
                    "ollama": version,
                    "model": model,
                    "show": details,
                },
            )
            raw_bytes = self._http("/api/chat", payload)
            # Save exact bytes even if JSON or the response schema is invalid.
            with (directory / "response.json").open("xb") as file:
                file.write(raw_bytes)
            raw = _object(raw_bytes)
            message = raw.get("message")
            if raw.get("done") is not True or not isinstance(message, dict):
                raise LLMCallError("Ollama did not return a completed chat response")
            text = message.get("content", "")
            if not isinstance(text, str):
                raise LLMCallError("Ollama returned non-text message content")
            if raw.get("model") != payload["model"]:
                raise LLMCallError("Ollama returned an unexpected model name")
            usage = _usage(raw)
            if self._model(payload["model"])["digest"] != model["digest"]:
                raise LLMCallError("Model digest changed during generation")
            result = LLMResponse(
                call_id,
                text,
                raw["model"],
                usage,
                time.monotonic() - started,
                raw,
            )
            _write_json(directory / "result.json", asdict(result))
            return result
        except LLMCallError as exc:
            known_usage = None
            if raw is not None:
                try:
                    known_usage = asdict(_usage(raw))
                except LLMCallError:
                    pass
            _write_json(
                directory / "error.json",
                {
                    "call_id": call_id,
                    "error": str(exc),
                    "duration_seconds": time.monotonic() - started,
                    "usage": known_usage,
                },
            )
            raise
