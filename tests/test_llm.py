import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from benchmark.contracts import LLMCallError, Message, ModelSettings
from benchmark.llm import OllamaClient


@pytest.fixture
def server():
    state = {
        "requests": [],
        "response": {
            "model": "qwen2.5-coder:7b-instruct-q4_K_M",
            "message": {"content": "---- MODULE Spec ----\n===="},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 30,
            "eval_count": 12,
        },
        "chat_status": 200,
        "digest": "sha256:test",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(
                {
                    "models": [
                        {"name": "qwen2.5-coder:7b-instruct-q4_K_M", "digest": state["digest"]}
                    ]
                }
                if self.path == "/api/tags"
                else {"version": "test"}
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            if self.path == "/api/chat":
                self.respond(state["response"], state["chat_status"])
            else:
                self.respond({"details": {"quantization_level": "Q4_K_M"}})

        def respond(self, body, status=200):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
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
    return ModelSettings(
        "ollama",
        "qwen2.5-coder:7b-instruct-q4_K_M",
        {
            "think": False,
            "keep_alive": 0,
            "options": {"num_ctx": 8192, "num_predict": 2048, "temperature": 0.7},
        },
    )


def test_real_http_roundtrip_records_provenance_and_keeps_chats_isolated(
    tmp_path, server, settings
):
    url, state = server
    client = OllamaClient(tmp_path, base_url=url)
    first = client.generate((Message("user", "first"),), settings)
    second = client.generate((Message("user", "second"),), settings)
    chats = [body for path, body in state["requests"] if path == "/api/chat"]
    assert chats[1]["messages"] == [{"role": "user", "content": "second"}]
    assert chats[0]["truncate"] is False and chats[0]["shift"] is False
    assert chats[0]["stream"] is False
    assert first.usage.input_tokens == 30 and first.usage.output_tokens == 12
    assert first.call_id != second.call_id
    directory = tmp_path / "calls" / first.call_id
    assert json.loads((directory / "request.json").read_text()) == chats[0]
    assert json.loads((directory / "response.json").read_text()) == state["response"]
    assert (
        json.loads((directory / "environment.json").read_text())["model"]["digest"] == "sha256:test"
    )
    assert (directory / "result.json").is_file()


@pytest.mark.parametrize(
    "change",
    [
        {"done": False},
        {"message": {"content": 17}},
        {"prompt_eval_count": -1},
        {"eval_count": True},
        {"model": "wrong-model"},
    ],
)
def test_invalid_responses_are_saved_and_reported_as_errors(tmp_path, server, settings, change):
    url, state = server
    state["response"].update(change)
    with pytest.raises(LLMCallError):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    directory = next((tmp_path / "calls").iterdir())
    assert (directory / "response.json").is_file()
    assert (directory / "error.json").is_file()
    assert not (directory / "result.json").exists()


def test_unknown_usage_is_preserved(tmp_path, server, settings):
    url, state = server
    del state["response"]["prompt_eval_count"]
    result = OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    assert result.usage.input_tokens is None


def test_http_error_is_recorded_without_implicit_retry(tmp_path, server, settings):
    url, state = server
    state.update(chat_status=500, response={"error": "out of memory"})
    with pytest.raises(LLMCallError, match="500"):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    assert len([p for p, _ in state["requests"] if p == "/api/chat"]) == 1
    directory = next((tmp_path / "calls").iterdir())
    assert "out of memory" in (directory / "error.json").read_text()


def test_digest_mismatch_stops_before_generation(tmp_path, server, settings):
    url, state = server
    with pytest.raises(LLMCallError, match="digest"):
        OllamaClient(tmp_path, base_url=url, expected_digest="other").generate(
            (Message("user", "hi"),),
            settings,
        )
    assert not state["requests"]


def test_bad_options_fail_before_any_io(tmp_path, settings):
    for parameters in ({"think": False}, {**settings.parameters, "truncate": True}):
        with pytest.raises(ValueError):
            OllamaClient(tmp_path).generate(
                (Message("user", "hi"),),
                replace(settings, parameters=parameters),
            )
    assert not list(tmp_path.iterdir())


def test_filesystem_failure_does_not_call_server(tmp_path, server, settings):
    url, state = server
    (tmp_path / "calls").write_text("not a directory")
    with pytest.raises(OSError):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    assert not state["requests"]


def test_output_limit_is_recorded_without_discarding_generated_text(tmp_path, server, settings):
    url, state = server
    state["response"]["done_reason"] = "length"
    result = OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    assert result.raw_response["done_reason"] == "length"
    assert result.text == state["response"]["message"]["content"]


def test_error_after_response_preserves_known_usage(tmp_path, server, settings):
    url, state = server
    state["response"]["model"] = "unexpected"
    with pytest.raises(LLMCallError):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    error = next((tmp_path / "calls").glob("*/error.json"))
    assert json.loads(error.read_text())["usage"] == {"input_tokens": 30, "output_tokens": 12}


def test_cli_saves_a_call_to_selected_artifact_root(tmp_path, server, capsys):
    from benchmark.cli import main

    url, _ = server
    messages = tmp_path / "messages.json"
    messages.write_text('[{"role":"user","content":"hi"}]')
    output = tmp_path / "output"
    assert (
        main(
            [
                "ollama-chat",
                "--messages",
                str(messages),
                "--output",
                str(output),
                "--base-url",
                url,
                "--seed",
                "42",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (output / "calls" / result["call_id"] / "response.json").is_file()


def test_cli_invalid_messages_does_not_start_a_call(tmp_path):
    from benchmark.cli import main

    messages = tmp_path / "messages.json"
    messages.write_text('{"role":"user","content":"hi"}')
    with pytest.raises(SystemExit) as exc:
        main(["ollama-chat", "--messages", str(messages), "--output", str(tmp_path / "out")])
    assert exc.value.code == 1
    assert not (tmp_path / "out").exists()


def test_malformed_json_is_preserved_verbatim(tmp_path, server, settings):
    url, state = server
    state["response"] = b"not-json\n"
    with pytest.raises(LLMCallError, match="invalid JSON"):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    response = next((tmp_path / "calls").glob("*/response.json"))
    assert response.read_bytes() == b"not-json\n"


def test_timeout_is_recorded_without_retry(tmp_path, settings, monkeypatch):
    client = OllamaClient(tmp_path, timeout_seconds=1)
    requests = []

    def timeout(*args, **kwargs):
        requests.append(args)
        raise TimeoutError("timed out")

    monkeypatch.setattr(client._opener, "open", timeout)
    with pytest.raises(LLMCallError, match="transport error"):
        client.generate((Message("user", "hi"),), settings)
    assert len(requests) == 1
    error = next((tmp_path / "calls").glob("*/error.json"))
    assert json.loads(error.read_text())["usage"] is None


def test_empty_completed_generation_is_returned_for_attempt_accounting(tmp_path, server, settings):
    url, state = server
    state["response"]["message"] = {"content": "", "thinking": "unfinished reasoning"}
    state["response"]["done_reason"] = "length"
    result = OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    assert result.text == ""
    assert result.usage.output_tokens == 12
    assert result.raw_response["message"]["thinking"] == "unfinished reasoning"


def test_backend_cannot_silently_exceed_requested_context(tmp_path, server, settings):
    url, state = server
    state["response"]["prompt_eval_count"] = 8190
    with pytest.raises(LLMCallError, match="context window"):
        OllamaClient(tmp_path, base_url=url).generate((Message("user", "hi"),), settings)
    error = next((tmp_path / "calls").glob("*/error.json"))
    assert json.loads(error.read_text())["usage"]["input_tokens"] == 8190
    assert (error.parent / "response.json").is_file()
    assert not (error.parent / "result.json").exists()
