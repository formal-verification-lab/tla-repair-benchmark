"""Recorded, isolated SANY/TLC execution using the shared verifier contract."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from benchmark.contracts import (
    CheckStatus,
    ToolResult,
    VerificationResult,
    VerifierSettings,
)

# Official tlc2.output.EC$ExitStatus constants in the pinned 1.7.4 JAR.
# Text rules retain their more specific labels; these cover unseen diagnostic
# wording within known specification-failure classes. Resource/system failures
# (152/153/255) deliberately remain outside this mapping.
TLC_SPEC_EXIT_STATUS = {
    10: "assumption",
    11: "deadlock",
    12: "safety",
    13: "temporal",
    14: "assertion",
    75: "evaluation",
    76: "safety_evaluation",
    77: "temporal_evaluation",
    150: "syntax",
    151: "configuration",
}


def _dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _run(command: tuple[str, ...], cwd: Path, timeout: float):
    started = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=cwd, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started, None
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or b"", exc.stderr or b"", time.monotonic() - started, "timeout"
    except OSError as exc:
        return None, b"", str(exc).encode(), time.monotonic() - started, "launch_error"


def _classify(tool: str, code: int | None, log: str, module: str):
    if re.search(
        r"OutOfMemoryError|Could not find or load main class|Exception in thread|"
        r"Unable to access jarfile|Could not create the Java Virtual Machine|"
        r"java\.rmi\.|java\.net\.|java\.io\.|StackOverflowError|NoClassDefFoundError",
        log,
    ):
        return CheckStatus.TOOL_ERROR, "runtime_error"
    if tool == "tlc" and code in {152, 153, 255}:
        return CheckStatus.TOOL_ERROR, {
            152: "state_space_limit",
            153: "runtime_error",
            255: "unrecognized_completion",
        }[code]
    if tool == "sany":
        if re.search(
            r"(?i)fatal errors|parse error|lexical error|semantic errors|"
            r"\*\*\*\s*(?:errors?|abort)",
            log,
        ):
            return CheckStatus.SPEC_ERROR, None
        if code == 0 and f"Semantic processing of module {module}" in log:
            return CheckStatus.PASSED, None
    else:
        if code == 0 and "Model checking completed. No error has been found." in log:
            return CheckStatus.PASSED, None
        # Only observed specification diagnostics count as specification failures.
        for pattern, category in (
            (r"Error: The constant parameter .+ is not assigned a value", "configuration"),
            (
                r"Error: The name \S+ specified in the configuration file\s+"
                r"is not defined in the specification\.",
                "configuration",
            ),
            (r"Error: Successor state is not completely specified", "incomplete_state"),
            (r"Invariant .+ is violated", "invariant"),
            (r"Deadlock reached", "deadlock"),
            (r"Temporal properties were violated", "temporal"),
            (
                r"Error: (?:The (?:behavior|initial|next|state)|In evaluation|Evaluating|"
                r"Attempted|TLC was unable|The invariant|The configuration)",
                None,
            ),
        ):
            if re.search(pattern, log):
                return CheckStatus.SPEC_ERROR, category or TLC_SPEC_EXIT_STATUS.get(code)
        if code in TLC_SPEC_EXIT_STATUS and re.search(r"^Error:", log, re.M):
            return CheckStatus.SPEC_ERROR, TLC_SPEC_EXIT_STATUS[code]
    return CheckStatus.TOOL_ERROR, "unrecognized_completion"


class JavaVerifier:
    """Create a fresh artifact directory; never overwrite a completed check."""

    def verify(
        self, spec_path: Path, cfg_path: Path, settings: VerifierSettings, work_dir: Path
    ) -> VerificationResult:
        for timeout in (settings.sany_timeout_seconds, settings.tlc_timeout_seconds):
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Timeouts must be finite and positive")
        # Restrict options to resource controls; no simulation, recovery or disabled checks.
        args = settings.tlc_arguments
        if len(args) % 2 or any(
            args[i] not in {"-workers", "-seed"} for i in range(0, len(args), 2)
        ):
            raise ValueError("Only explicit -workers and -seed pairs are supported")
        for i in range(0, len(args), 2):
            if not args[i + 1].isdigit() or (args[i] == "-workers" and int(args[i + 1]) < 1):
                raise ValueError("Invalid TLC resource option")
        jar = settings.jar_path.resolve(strict=True)
        digest = hashlib.sha256(jar.read_bytes()).hexdigest()
        if digest != settings.jar_sha256:
            raise ValueError("JAR SHA-256 mismatch")
        spec_path = spec_path.resolve(strict=True)
        cfg_path = cfg_path.resolve(strict=True)
        if spec_path.suffix != ".tla" or not re.fullmatch(
            r"[A-Za-z_][A-Za-z_0-9]*", spec_path.stem
        ):
            raise ValueError("Specification must be a named .tla module")
        source, config = spec_path.read_bytes(), cfg_path.read_bytes()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=False)
        (work_dir / spec_path.name).write_bytes(source)
        (work_dir / "Spec.cfg").write_bytes(config)
        java = settings.java_executable
        if "/" in java:
            java = str(Path(java).resolve())
        version = _run((java, "-version"), work_dir, settings.sany_timeout_seconds)
        tools_version = _run(
            (java, "-cp", str(jar), "tlc2.TLC", "-help"), work_dir, settings.sany_timeout_seconds
        )
        _dump(
            work_dir / "environment.json",
            {
                "settings": {**asdict(settings), "jar_path": str(jar)},
                "jar_sha256": digest,
                "spec_sha256": hashlib.sha256(source).hexdigest(),
                "cfg_sha256": hashlib.sha256(config).hexdigest(),
                "tools_version": {
                    "command": [java, "-cp", str(jar), "tlc2.TLC", "-help"],
                    "stdout": tools_version[1].decode(errors="replace"),
                    "stderr": tools_version[2].decode(errors="replace"),
                    "exit_code": tools_version[0],
                    "error": tools_version[4],
                },
                "java_version": {
                    "exit_code": version[0],
                    "stdout": version[1].decode(errors="replace"),
                    "stderr": version[2].decode(errors="replace"),
                    "error": version[4],
                },
            },
        )
        results = []
        for tool in ("sany", "tlc"):
            if tool == "tlc" and results[0].status != CheckStatus.PASSED:
                results.append(ToolResult("tlc", CheckStatus.NOT_RUN, (), "", "", None, 0))
                break
            command = (java, "-Xmx512m", "-XX:+UseSerialGC", "-cp", str(jar))
            if tool == "sany":
                command += ("tla2sany.SANY", spec_path.name)
                timeout = settings.sany_timeout_seconds
            else:
                command += (
                    "tlc2.TLC",
                    "-workers",
                    "1",
                    *args,
                    "-metadir",
                    "states",
                    "-config",
                    "Spec.cfg",
                    spec_path.name,
                )
                timeout = settings.tlc_timeout_seconds
            code, stdout, stderr, elapsed, error = _run(command, work_dir, timeout)
            (work_dir / f"{tool}.stdout.log").write_bytes(stdout)
            (work_dir / f"{tool}.stderr.log").write_bytes(stderr)
            out, err = stdout.decode(errors="replace"), stderr.decode(errors="replace")
            if error:
                status = CheckStatus.TIMEOUT if error == "timeout" else CheckStatus.TOOL_ERROR
                category = error
            else:
                status, category = _classify(tool, code, out + "\n" + err, spec_path.stem)
            result = ToolResult(tool, status, command, out, err, code, elapsed, category)
            results.append(result)
            _dump(work_dir / f"{tool}.json", asdict(result))
        verification = VerificationResult(*results)
        _dump(work_dir / "verification.json", asdict(verification))
        return verification


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--jar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--sany-timeout", type=float, default=30)
    parser.add_argument("--tlc-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    settings = VerifierSettings(
        args.jar, args.jar_sha256, args.java, args.sany_timeout, args.tlc_timeout
    )
    result = JavaVerifier().verify(args.spec, args.cfg, settings, args.output)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
