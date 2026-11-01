"""Inspect contracts and make a recorded local model call."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from benchmark import SCHEMA_VERSION, __version__
from benchmark.contracts import CheckStatus, LLMCallError, Message, Method, ModelSettings
from benchmark.llm import OllamaClient


def _messages(path: Path) -> tuple[Message, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Messages file must contain a nonempty JSON array")
    return tuple(Message(**entry) for entry in value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLA+ repair benchmark")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("schema-info", help="Print shared schema version and enum values")
    chat = subparsers.add_parser("ollama-chat", help="Make one recorded local Ollama call")
    chat.add_argument(
        "--messages", type=Path, required=True, help="JSON array of role/content messages"
    )
    chat.add_argument("--output", type=Path, required=True, help="Experiment artifact root")
    chat.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M")
    chat.add_argument("--base-url", default="http://127.0.0.1:11434")
    chat.add_argument("--timeout", type=float, default=600)
    chat.add_argument("--context-window", type=int, default=8192)
    chat.add_argument("--max-output", type=int, default=2048)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    chat.add_argument("--seed", type=int)
    chat.add_argument("--expected-digest")
    args = parser.parse_args(argv)
    if args.command == "schema-info":
        result = {
            "schema_version": SCHEMA_VERSION,
            "methods": {method.value: method.name for method in Method},
            "check_statuses": [status.value for status in CheckStatus],
        }
    else:
        try:
            options = {
                "num_ctx": args.context_window,
                "num_predict": args.max_output,
                "temperature": args.temperature,
            }
            if args.seed is not None:
                options["seed"] = args.seed
            settings = ModelSettings(
                "ollama",
                args.model,
                {
                    "think": args.think,
                    "keep_alive": 0,
                    "options": options,
                },
            )
            client = OllamaClient(
                args.output,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
                expected_digest=args.expected_digest,
            )
            result = asdict(client.generate(_messages(args.messages), settings))
            # Full raw response is already stored on disk.
            del result["raw_response"]
        except (OSError, ValueError, TypeError, LLMCallError) as exc:
            parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
