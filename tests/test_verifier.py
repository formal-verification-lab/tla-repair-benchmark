import hashlib
import os
from pathlib import Path

import pytest

from benchmark.contracts import CheckStatus, VerifierSettings
from benchmark.verifier import JavaVerifier, _classify, _run

SOURCE = """---- MODULE Sample ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x' = 1 - x
Inv == x \\in {0, 1}
====
"""
CFG = "INIT Init\nNEXT Next\nINVARIANT Inv\n"


@pytest.mark.parametrize(
    "message,category",
    [
        (
            "Error: The constant parameter N is not assigned a value by the configuration file.",
            "configuration",
        ),
        (
            "Error: Successor state is not completely specified by the next-state action.",
            "incomplete_state",
        ),
    ],
)
def test_model_errors_are_not_classified_as_verifier_infrastructure(message, category):
    assert _classify("tlc", 75, message, "Sample") == (CheckStatus.SPEC_ERROR, category)


@pytest.fixture
def inputs(tmp_path):
    spec, cfg = tmp_path / "Sample.tla", tmp_path / "Sample.cfg"
    spec.write_text(SOURCE)
    cfg.write_text(CFG)
    return spec, cfg


@pytest.mark.parametrize(
    "tool,code,log,status",
    [
        ("sany", 0, "Semantic processing of module Sample\n*** Errors: 1", "spec_error"),
        ("sany", 0, "Parsing file Sample.tla", "tool_error"),
        ("sany", 1, "Semantic processing of module Sample", "tool_error"),
        ("tlc", 0, "Finished in 1s", "tool_error"),
        ("tlc", 12, "Error: Invariant Inv is violated.", "spec_error"),
        ("tlc", 1, "java.lang.OutOfMemoryError", "tool_error"),
        (
            "tlc",
            1,
            "Error: TLC threw an unexpected exception.\njava.rmi.server.ExportException",
            "tool_error",
        ),
        ("tlc", 1, "Error: unknown condition", "tool_error"),
        ("tlc", 0, "Model checking completed. No error has been found.", "passed"),
    ],
)
def test_completion_requires_evidence(tool, code, log, status):
    assert _classify(tool, code, log, "Sample")[0] == status


def test_timeout_preserves_partial_output(tmp_path):
    import sys

    result = _run(
        (sys.executable, "-c", 'import time; print("partial", flush=True); time.sleep(5)'),
        tmp_path,
        0.1,
    )
    assert result[0] is None
    assert result[1] == b"partial\n"
    assert result[4] == "timeout"


def test_missing_java_and_no_overwrite(tmp_path, inputs):
    jar = tmp_path / "tools.jar"
    jar.write_bytes(b"fixture")
    settings = VerifierSettings(jar, hashlib.sha256(b"fixture").hexdigest(), "/missing/java", 1, 1)
    spec, cfg = inputs
    output = tmp_path / "check"
    result = JavaVerifier().verify(spec, cfg, settings, output)
    assert result.sany.status == CheckStatus.TOOL_ERROR
    assert result.tlc.status == CheckStatus.NOT_RUN
    assert (output / "verification.json").exists()
    with pytest.raises(FileExistsError):
        JavaVerifier().verify(spec, cfg, settings, output)


def test_hash_and_unsafe_options_rejected_before_execution(tmp_path, inputs):
    jar = tmp_path / "tools.jar"
    jar.write_bytes(b"fixture")
    spec, cfg = inputs
    settings = VerifierSettings(jar, "incorrect", "java", 1, 1)
    with pytest.raises(ValueError, match="SHA-256"):
        JavaVerifier().verify(spec, cfg, settings, tmp_path / "check")
    settings = VerifierSettings(jar, "incorrect", "java", 1, 1, ("-deadlock",))
    with pytest.raises(ValueError, match="Only explicit"):
        JavaVerifier().verify(spec, cfg, settings, tmp_path / "check")
    assert not (tmp_path / "check").exists()


@pytest.mark.skipif(not os.environ.get("TLA_TOOLS_JAR"), reason="Set TLA_TOOLS_JAR for Java checks")
@pytest.mark.parametrize(
    "case",
    ["valid", "syntax", "semantic", "invariant", "deadlock", "timeout", "incomplete", "temporal"],
)
def test_real_java(tmp_path, inputs, case):
    jar = Path(os.environ["TLA_TOOLS_JAR"])
    # Require the same official 1.7.4 binary used by task preparation.
    digest = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
    spec, cfg = inputs
    source = SOURCE
    if case == "syntax":
        source = source.replace("Init == x = 0", "Init == x =")
    if case == "semantic":
        source = source.replace("Init == x = 0", "Init == x = UnknownSymbol")
    if case == "invariant":
        source = source.replace("Inv == x \\in {0, 1}", "Inv == x = 0")
    if case == "deadlock":
        source = source.replace("Next == x' = 1 - x", "Next == x = 0 /\\ x' = 1")
    if case == "incomplete":
        source = source.replace("Init == x = 0", "Init == TRUE")
    if case == "temporal":
        source = source.replace("====", "Prop == <> (x = 2)\n====")
        cfg.write_text(CFG + "PROPERTY Prop\n")
    spec.write_text(source)
    settings = VerifierSettings(jar, digest, "java", 10, 0.001 if case == "timeout" else 10)
    result = JavaVerifier().verify(spec, cfg, settings, tmp_path / "check")
    if case in {"syntax", "semantic"}:
        assert result.sany.status == CheckStatus.SPEC_ERROR
        assert result.tlc.status == CheckStatus.NOT_RUN
    elif case == "timeout":
        assert result.tlc.status == CheckStatus.TIMEOUT
    else:
        assert result.sany.status == CheckStatus.PASSED
        assert result.tlc.status == (
            CheckStatus.PASSED if case == "valid" else CheckStatus.SPEC_ERROR
        )
    assert result.success == (case == "valid")
