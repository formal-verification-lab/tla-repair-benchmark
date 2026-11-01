"""Provider-independent contracts. No network, Java, or Gold reads on import."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Mapping, Protocol

from benchmark import SCHEMA_VERSION


class Method(StrEnum):
    FULL_LOG = "A"
    FILTERED_LOG = "B"
    LLM_EXPLANATION = "C"


class CheckStatus(StrEnum):
    PASSED = "passed"
    SPEC_ERROR = "spec_error"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    # None means unavailable; zero means a measured zero.
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and value < 0:
                raise ValueError("Token counts cannot be negative")


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    # Explicit parameters only; unsupported parameters must not be silently sent.
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class LLMResponse:
    call_id: str
    text: str
    model: str
    usage: Usage
    duration_seconds: float
    # Provider response with credentials/headers removed, preserved by caller.
    raw_response: Mapping[str, object]


class LLMCallError(RuntimeError):
    """Technical API failure, not an invalid specification or completed attempt."""


class LLMClient(Protocol):
    def generate(self, messages: tuple[Message, ...], settings: ModelSettings) -> LLMResponse:
        """Return a completed response or raise LLMCallError."""
        ...


@dataclass(frozen=True)
class TaskInput:
    """Only these task fields may enter generation/repair prompts; no Gold path."""

    task_id: str
    module_name: str
    description: str
    cfg_text: str


@dataclass(frozen=True)
class TaskMetadata:
    task_id: str
    difficulty: Literal["easy", "medium"]
    source_url: str
    source_commit: str
    source_task_path: str
    description_sha256: str
    cfg_sha256: str
    gold_sha256: str
    # Source files and difficulty must be recorded without inventing provenance.


@dataclass(frozen=True)
class VerifierSettings:
    jar_path: Path
    jar_sha256: str
    java_executable: str
    sany_timeout_seconds: float
    tlc_timeout_seconds: float
    tlc_arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    tool: Literal["sany", "tlc"]
    status: CheckStatus
    command: tuple[str, ...]
    stdout: str
    stderr: str
    exit_code: int | None
    duration_seconds: float
    error_type: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    sany: ToolResult
    tlc: ToolResult

    def __post_init__(self) -> None:
        if self.sany.tool != "sany" or self.tlc.tool != "tlc":
            raise ValueError("Verification results must be ordered SANY then TLC")
        if self.sany.status != CheckStatus.PASSED and self.tlc.status != CheckStatus.NOT_RUN:
            raise ValueError("TLC must not run before SANY passes")

    @property
    def success(self) -> bool:
        return self.sany.status == self.tlc.status == CheckStatus.PASSED


class Verifier(Protocol):
    def verify(
        self, spec_path: Path, cfg_path: Path, settings: VerifierSettings, work_dir: Path
    ) -> VerificationResult:
        """Run SANY, then TLC only after SANY success, in an isolated work directory."""
        ...


@dataclass(frozen=True)
class FeedbackResult:
    method: Method
    text: str
    explanation_call_id: str | None = None


class FeedbackStrategy(Protocol):
    def build(self, specification: str, verification: VerificationResult) -> FeedbackResult:
        """Preserve raw logs; explanation strategy receives its client at construction."""
        ...


@dataclass(frozen=True)
class CallRecord:
    """One actual API response, stored once even if referenced by three branches."""

    call_id: str
    purpose: Literal["initial", "repair", "explanation"]
    task_id: str
    repetition: int
    method: Method | None
    request_path: str
    response_path: str
    usage: Usage
    duration_seconds: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.repetition < 1:
            raise ValueError("Repetitions start at 1")
        if (self.purpose == "initial") != (self.method is None):
            raise ValueError("Only shared initial calls have no method")


@dataclass(frozen=True)
class AttemptRecord:
    task_id: str
    repetition: int
    method: Method
    iteration: int
    generation_call_id: str
    specification_path: str
    verification_path: str
    feedback_path: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 1 <= self.iteration <= 10 or self.repetition < 1:
            raise ValueError("Iteration must be 1..10; repetition must be positive")
