"""Recorded GigaChat API calls. OAuth secrets remain only in process memory."""

import json
import math
import ssl
import time
from dataclasses import asdict
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from benchmark.contracts import LLMCallError, LLMResponse, Message, ModelSettings, Usage
from benchmark.llm import _write_json


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GigaChatError(LLMCallError):
    def __init__(self, message, *, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _url(value, allowed):
    parsed = urlsplit(value)
    local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        not (local or value.rstrip("/") in allowed)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an official HTTPS endpoint or a local HTTP test server")
    return value.rstrip("/")


def _object(data):
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise GigaChatError("GigaChat returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise GigaChatError("GigaChat response must be an object")
    return result


class GigaChatClient:
    """Stateless chat histories; no hidden generation retries or paid fallback.

    OAuth refresh before a call is independent of a generation attempt. Access
    tokens, authorization keys, and OAuth responses are never written to disk.
    HTTPS verification stays enabled; an additional CA bundle is application scoped.
    """

    def __init__(
        self,
        artifact_root: Path,
        *,
        credentials: str,
        ca_bundle_file: Path | None = None,
        scope: str = "GIGACHAT_API_PERS",
        base_url: str = "https://api.giga.chat/v1",
        oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        timeout_seconds: float = 120,
        expected_model: str | None = None,
    ):
        if not credentials or any(c.isspace() for c in credentials):
            raise ValueError("Provide the Base64 authorization key without the Basic prefix")
        if scope != "GIGACHAT_API_PERS":
            raise ValueError("This client is configured for the personal API scope")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.artifact_root = Path(artifact_root)
        self.base_url = _url(
            base_url, {"https://api.giga.chat/v1", "https://gigachat.devices.sberbank.ru/api/v1"}
        )
        self.oauth_url = _url(oauth_url, {"https://ngw.devices.sberbank.ru:9443/api/v2/oauth"})
        self.scope = scope
        self.timeout = timeout_seconds
        self.expected_model = expected_model
        self._credentials = credentials
        self._token = None
        self._expires = 0
        context = ssl.create_default_context()
        if ca_bundle_file:
            context.load_verify_locations(cafile=str(ca_bundle_file))
        self._opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), _NoRedirect())
        self.ca_bundle_file = Path(ca_bundle_file) if ca_bundle_file else None

    def _safe(self, value):
        if isinstance(value, str):
            for secret in (self._credentials, self._token):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._safe(k): self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        return value

    def _write(self, path, value):
        _write_json(path, self._safe(value))

    def _http(self, url, headers, data=None):
        try:
            with self._opener.open(
                Request(url, data=data, headers=headers), timeout=self.timeout
            ) as response:
                return response.read()
        except HTTPError as exc:
            # Provider bodies can echo credentials. Retain safe diagnostics, never
            # the OAuth body or untrusted headers, and never retry the generation.
            raise GigaChatError(
                f"GigaChat HTTP {exc.code}",
                status=exc.code,
                retry_after=self._safe(exc.headers.get("Retry-After")),
            ) from None
        except (URLError, OSError, HTTPException) as exc:
            raise GigaChatError(self._safe(f"GigaChat transport error: {exc}")) from None

    def _authorize(self):
        if self._token and time.time() < self._expires - 30:
            return
        raw = _object(
            self._http(
                self.oauth_url,
                {
                    "Authorization": "Basic " + self._credentials,
                    "RqUID": str(uuid4()),
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                urlencode({"scope": self.scope}).encode(),
            )
        )
        token, expiry = raw.get("access_token"), raw.get("expires_at")
        if not isinstance(token, str) or not token or type(expiry) is not int:
            raise GigaChatError("Invalid OAuth response")
        self._token = token
        self._expires = expiry / 1000
        if self._expires <= time.time():
            raise GigaChatError("OAuth returned an expired token")

    def _api(self, path, payload=None):
        self._authorize()
        return self._http(
            self.base_url + path,
            {
                "Authorization": "Bearer " + self._token,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            if payload is not None
            else None,
        )

    def _metadata(self):
        import hashlib

        return {
            "provider": "gigachat",
            "base_url": self.base_url,
            "oauth_url": self.oauth_url,
            "scope": self.scope,
            "timeout_seconds": self.timeout,
            "expected_model": self.expected_model,
            "usage_semantics": (
                "input_tokens includes precached_prompt_tokens; raw total_tokens is billable"
            ),
            "tls_verification": True,
            "ca_bundle_sha256": hashlib.sha256(self.ca_bundle_file.read_bytes()).hexdigest()
            if self.ca_bundle_file
            else None,
            "started_at": datetime.now(UTC).isoformat(),
        }

    def access(self):
        directory = self.artifact_root / "access" / uuid4().hex
        directory.mkdir(parents=True)
        self._write(directory / "client.json", self._metadata())
        try:
            result = {
                "models": _object(self._api("/models")),
                "balance": _object(self._api("/balance")),
            }
            self._write(directory / "result.json", result)
            return self._safe({"artifact_directory": str(directory), **result})
        except GigaChatError as exc:
            self._write(
                directory / "error.json",
                {"error": str(exc), "http_status": exc.status, "retry_after": exc.retry_after},
            )
            raise

    def generate(self, messages: tuple[Message, ...], settings: ModelSettings) -> LLMResponse:
        if settings.provider != "gigachat" or not settings.model.strip():
            raise ValueError("Use provider='gigachat' and an explicit model ID")
        if not messages or any(
            m.role not in {"system", "user", "assistant"} or not isinstance(m.content, str)
            for m in messages
        ):
            raise ValueError("Provide nonempty messages with supported roles and text content")
        params = dict(settings.parameters)
        if params.keys() - {"max_tokens", "temperature", "top_p", "repetition_penalty"}:
            raise ValueError("Unsupported GigaChat generation parameter")
        if type(params.get("max_tokens")) is not int or params["max_tokens"] <= 0:
            raise ValueError("Set an explicit positive max_tokens")
        payload = json.loads(
            json.dumps(
                {
                    "model": settings.model,
                    "messages": [asdict(m) for m in messages],
                    **params,
                    "stream": False,
                    "function_call": "none",
                },
                allow_nan=False,
            )
        )
        call_id = uuid4().hex
        directory = self.artifact_root / "calls" / call_id
        directory.mkdir(parents=True)
        self._write(directory / "client.json", {"call_id": call_id, **self._metadata()})
        self._write(directory / "request.json", payload)
        started = time.monotonic()
        usage = Usage()
        try:
            raw_bytes = self._api("/chat/completions", payload)
            # Preserve exact bytes except for defensive credential redaction.
            for secret in (self._credentials, self._token):
                raw_bytes = raw_bytes.replace(secret.encode(), b"[REDACTED]")
            (directory / "response.json").write_bytes(raw_bytes)
            raw = _object(raw_bytes)
            counts = raw.get("usage", {})
            if not isinstance(counts, dict):
                raise GigaChatError("Invalid GigaChat usage")
            values = [counts.get(k) for k in ("prompt_tokens", "completion_tokens")]
            if any(v is not None and (type(v) is not int or v < 0) for v in values):
                raise GigaChatError("Invalid GigaChat token count")
            cached = counts.get("precached_prompt_tokens", 0)
            if type(cached) is not int or cached < 0:
                raise GigaChatError("Invalid GigaChat cached token count")
            # GigaChat prompt_tokens excludes cache hits (unlike some other
            # providers). Normalize full context usage; keep billing fields raw.
            if values[0] is not None:
                values[0] += cached
            usage = Usage(*values)
            choices = raw.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise GigaChatError("Expected one completed GigaChat response")
            choice = choices[0]
            message = choice.get("message")
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("content"), str)
                or not isinstance(choice.get("finish_reason"), str)
            ):
                raise GigaChatError("Invalid GigaChat completion")
            if not isinstance(raw.get("model"), str) or not raw["model"]:
                raise GigaChatError("Missing resolved GigaChat model ID")
            if self.expected_model is not None and raw["model"] != self.expected_model:
                raise GigaChatError("Resolved GigaChat model differs from expected_model")
            # Provider may return a versioned model ID instead of the requested
            # family alias. Keep both; the runner must freeze the observed version.
            result = LLMResponse(
                call_id, message["content"], raw["model"], usage, time.monotonic() - started, raw
            )
            self._write(directory / "result.json", asdict(result))
            return result
        except GigaChatError as exc:
            self._write(
                directory / "error.json",
                {
                    "call_id": call_id,
                    "error": str(exc),
                    "http_status": exc.status,
                    "retry_after": exc.retry_after,
                    "duration_seconds": time.monotonic() - started,
                    "usage": asdict(usage),
                },
            )
            raise
