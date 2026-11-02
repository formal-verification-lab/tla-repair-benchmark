import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from benchmark.contracts import Message, ModelSettings
from benchmark.gigachat import GigaChatClient, GigaChatError


@pytest.fixture
def server():
    state = {
        "requests": [],
        "chat_status": 200,
        "response": {
            "model": "GigaChat-3-Ultra:test-version",
            "choices": [
                {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append((self.path, None, dict(self.headers)))
            self.respond(
                {"data": [{"id": "GigaChat-3-Ultra"}]}
                if self.path.endswith("models")
                else {"balance": [{"usage": "GigaChat-Ultra", "value": 50000000}]}
            )

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"].append((self.path, body, dict(self.headers)))
            if self.path == "/oauth":
                self.respond(
                    {
                        "access_token": "secret-access-token",
                        "expires_at": int((time.time() + 1800) * 1000),
                    }
                )
            else:
                self.respond(state["response"], state["chat_status"])

        def respond(self, body, status=200):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Retry-After", "60")
            if status == 302:
                self.send_header("Location", "/must-not-forward-credentials")
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.fixture
def settings():
    return ModelSettings("gigachat", "GigaChat-3-Ultra", {"max_tokens": 256, "temperature": 0.1})


def client(tmp_path, url, **kwargs):
    return GigaChatClient(
        tmp_path,
        credentials="secret-authorization-key",
        base_url=url + "/v1",
        oauth_url=url + "/oauth",
        **kwargs,
    )


def test_real_http_oauth_cache_and_isolated_histories(tmp_path, server, settings):
    url, state = server
    giga = client(tmp_path, url)
    first = giga.generate((Message("user", "first"),), settings)
    giga.generate((Message("user", "second"),), settings)
    oauth = [r for r in state["requests"] if r[0] == "/oauth"]
    chats = [r for r in state["requests"] if r[0].endswith("completions")]
    assert len(oauth) == 1 and len(chats) == 2
    assert oauth[0][1] == b"scope=GIGACHAT_API_PERS"
    assert oauth[0][2]["Authorization"] == "Basic secret-authorization-key"
    assert chats[0][2]["Authorization"] == "Bearer secret-access-token"
    assert json.loads(chats[1][1])["messages"] == [{"role": "user", "content": "second"}]
    assert json.loads(chats[0][1])["function_call"] == "none"
    assert first.model == "GigaChat-3-Ultra:test-version" and first.usage.input_tokens == 8
    saved = tmp_path / "calls" / first.call_id
    assert json.loads((saved / "response.json").read_text()) == state["response"]
    assert json.loads((saved / "request.json").read_text())["model"] == "GigaChat-3-Ultra"
    assert not any(
        "secret-access-token" in p.read_text() or "secret-authorization-key" in p.read_text()
        for p in tmp_path.rglob("*.json")
    )


def test_rate_limit_is_recorded_without_generation_retry(tmp_path, server, settings):
    url, state = server
    state.update(
        chat_status=429, response={"error": "secret-access-token secret-authorization-key"}
    )
    with pytest.raises(GigaChatError, match="429"):
        client(tmp_path, url).generate((Message("user", "test"),), settings)
    assert len([r for r in state["requests"] if r[0].endswith("completions")]) == 1
    error = json.loads(next(tmp_path.rglob("error.json")).read_text())
    assert error["http_status"] == 429 and error["retry_after"] == "60"
    assert "secret-" not in json.dumps(error)


def test_input_usage_includes_cached_context_while_raw_billing_is_preserved(
    tmp_path, server, settings
):
    url, state = server
    state["response"]["usage"].update(precached_prompt_tokens=3, total_tokens=12)
    result = client(tmp_path, url).generate((Message("user", "hi"),), settings)
    assert result.usage.input_tokens == 11 and result.usage.output_tokens == 4
    assert result.raw_response["usage"]["total_tokens"] == 12


def test_model_drift_keeps_response_and_usage_but_rejects_result(tmp_path, server, settings):
    url, _ = server
    with pytest.raises(GigaChatError, match="expected_model"):
        client(tmp_path, url, expected_model="previous-version").generate(
            (Message("user", "hi"),), settings
        )
    assert next(tmp_path.rglob("response.json")).exists()
    error = json.loads(next(tmp_path.rglob("error.json")).read_text())
    assert error["usage"] == {"input_tokens": 8, "output_tokens": 4}
    assert not list(tmp_path.rglob("result.json"))


@pytest.mark.parametrize(
    "field,value", [("choices", []), ("usage", {"prompt_tokens": True}), ("model", None)]
)
def test_invalid_responses_are_preserved(tmp_path, server, settings, field, value):
    url, state = server
    state["response"][field] = value
    with pytest.raises(GigaChatError):
        client(tmp_path, url).generate((Message("user", "hi"),), settings)
    assert next(tmp_path.rglob("response.json")).exists()
    assert next(tmp_path.rglob("error.json")).exists()


def test_access_records_models_and_quota_without_oauth_secrets(tmp_path, server):
    url, _ = server
    result = client(tmp_path, url).access()
    assert result["balance"]["balance"][0]["value"] == 50000000
    assert result["models"]["data"][0]["id"] == "GigaChat-3-Ultra"
    assert "secret-" not in next(tmp_path.rglob("result.json")).read_text()


def test_defensive_redaction_covers_server_echo_and_request_messages(tmp_path, server, settings):
    url, state = server
    state["response"]["debug"] = "secret-authorization-key secret-access-token"
    client(tmp_path, url).generate((Message("user", "secret-authorization-key"),), settings)
    assert not any(
        "secret-authorization-key" in p.read_text() or "secret-access-token" in p.read_text()
        for p in tmp_path.rglob("*.json")
    )


def test_redirect_cannot_forward_authorization(tmp_path, server, settings):
    url, state = server
    state["chat_status"] = 302
    with pytest.raises(GigaChatError, match="302"):
        client(tmp_path, url).generate((Message("user", "hi"),), settings)
    assert len(state["requests"]) == 2


def test_expired_oauth_is_refreshed_before_next_call(tmp_path, server, settings):
    url, state = server
    giga = client(tmp_path, url)
    giga.generate((Message("user", "first"),), settings)
    giga._expires = 0
    giga.generate((Message("user", "second"),), settings)
    assert len([r for r in state["requests"] if r[0] == "/oauth"]) == 2
    assert len([r for r in state["requests"] if r[0].endswith("completions")]) == 2


def test_cli_missing_credentials_does_not_start_a_call(tmp_path, monkeypatch, capsys):
    from benchmark.cli import main

    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["gigachat-access", "--output", str(tmp_path / "artifacts")])
    assert exc.value.code == 1
    assert "credentials-file" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()


def test_cli_invalid_messages_are_rejected_before_authorization(tmp_path, monkeypatch):
    from benchmark.cli import main

    messages = tmp_path / "messages.json"
    messages.write_text('{"role":"user","content":"invalid root"}')
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "secret-authorization-key")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "gigachat-chat",
                "--model",
                "GigaChat-3-Ultra",
                "--messages",
                str(messages),
                "--output",
                str(tmp_path / "artifacts"),
            ]
        )
    assert exc.value.code == 1
    assert not (tmp_path / "artifacts").exists()
