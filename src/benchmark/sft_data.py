"""Synthetic, tool-validated SFT data. Never reads coursework tasks or their Gold.

This validator is scoped to preparing synthetic training fixtures. The main
benchmark verifier and feedback strategies remain independent components.
"""

import hashlib
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM = "Write executable TLA+ source only. Respect the task and the fixed TLC configuration."
EXPLAIN_SYSTEM = "Explain the TLA+ diagnostic in plain text, without code or a rewritten module."
CFG = "SPECIFICATION Spec\nINVARIANT TypeOK\n"


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    split: str
    description: str
    correct: str
    broken: str
    explanation: str
    expected_states: int
    defect: str
    cfg: str = CFG


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def make_case(family: str, n: int, split: str, defect_index: int) -> Case:
    name = family.title().replace("_", "") + str(n)
    extends = "EXTENDS Naturals\n"
    variables, initial, invariant, states = "x", "x = 0", f"x \\in 0..{n}", n + 1
    if family == "modulo":
        next_action = f"x' = (x + 1) % {n + 1}"
        description = f"x starts at 0 and cycles through 0..{n}, incrementing and wrapping to 0."
    elif family == "toggle":
        variables = "flag"
        initial = "flag = FALSE" if n % 2 == 0 else "flag = TRUE"
        invariant = "flag \\in BOOLEAN"
        next_action = "flag' = ~flag"
        states = 2
        description = f"A Boolean flag starts with {initial}. Every step flips flag."
    elif family == "resource":
        next_action = f"(x < {n} /\\ x' = x + 1) \\/ (x > 0 /\\ x' = x - 1)"
        description = (
            f"Occupancy x starts at 0. Allocate increments below capacity {n}; "
            f"release decrements above 0."
        )
    elif family == "saturating":
        next_action = f"x' = IF x < {n} THEN x + 1 ELSE x"
        description = f"x starts at 0 and increments up to {n}, then stays at {n}."
    elif family == "pool":
        variables, initial, invariant = "pending", f"pending = 1..{n}", f"pending \\subseteq 1..{n}"
        next_action = (
            f"(\\E i \\in pending : pending' = pending \\ {{i}}) \\/ (pending = "
            f"{{}} /\\ pending' = 1..{n})"
        )
        states = 2**n
        description = (
            f"pending starts as all jobs 1..{n}. Remove any one pending job per "
            f"step; reset to all jobs when empty."
        )
    elif family in {"fifo", "stack"}:
        extends = "EXTENDS Naturals, Sequences\n"
        variables, initial = "q", "q = <<>>"
        invariant = f"q \\in Seq({{0, 1}}) /\\ Len(q) <= {n}"
        pop = "Tail(q)" if family == "fifo" else "SubSeq(q, 1, Len(q) - 1)"
        next_action = (
            f"(Len(q) < {n} /\\ \\E v \\in {{0, 1}} : q' = Append(q, v)) \\/ "
            f"(Len(q) > 0 /\\ q' = {pop})"
        )
        states = 2 ** (n + 1) - 1
        end = "first" if family == "fifo" else "last"
        description = (
            f"q is initially an empty sequence. Append any bit when length is "
            f"below {n}, or remove the {end} element when nonempty."
        )
    elif family == "phase":
        variables, initial = "phase", 'phase = "p0"'
        values = ", ".join(f'"p{i}"' for i in range(n))
        invariant = f"phase \\in {{{values}}}"
        next_action = " \\/ ".join(
            f'(phase = "p{i}" /\\ phase\' = "p{(i + 1) % n}")' for i in range(n)
        )
        states = n
        description = (
            f"phase starts at p0 and follows the cycle p0,...,p{n - 1},p0. Phases are strings."
        )
    elif family == "pair":
        variables, initial = "x, y", "x = 0 /\\ y = 0"
        invariant = f"x \\in 0..{n} /\\ y \\in 0..{n}"
        next_action = (
            f"(x' = (x + 1) % {n + 1} /\\ UNCHANGED y) \\/ (y' = (y + 1) % {n + 1} /\\ UNCHANGED x)"
        )
        states = (n + 1) ** 2
        description = (
            f"x and y start at 0. Each step increments either x or y modulo "
            f"{n + 1}, preserving the other."
        )
    elif family == "flags":
        variables = "a, b"
        initial = "a = FALSE /\\ b = FALSE" if n == 2 else "a = TRUE /\\ b = TRUE"
        invariant = "a \\in BOOLEAN /\\ b \\in BOOLEAN"
        next_action = "(a' = ~a /\\ UNCHANGED b) \\/ (b' = ~b /\\ UNCHANGED a)"
        states = 4
        description = (
            f"Two Boolean variables satisfy initially {initial}. Each step flips "
            f"exactly one and preserves the other."
        )
    elif family == "mutex":
        variables, initial = "owner", "owner = 0"
        invariant = f"owner \\in 0..{n}"
        next_action = f"(owner = 0 /\\ owner' \\in 1..{n}) \\/ (owner # 0 /\\ owner' = 0)"
        description = (
            f"owner starts at 0, meaning free. Any process in 1..{n} can acquire "
            f"only when free. The owner can release back to 0."
        )
    elif family == "handshake":
        variables = "req, ack"
        initial = "req = FALSE /\\ ack = FALSE" if n == 2 else "req = TRUE /\\ ack = TRUE"
        invariant = "req \\in BOOLEAN /\\ ack \\in BOOLEAN"
        next_action = (
            "(req = ack /\\ req' = ~req /\\ UNCHANGED ack) \\/ (req # ack /\\ "
            "ack' = req /\\ UNCHANGED req)"
        )
        states = 4
        description = (
            f"Initially {initial}. When req equals ack, the sender flips req "
            f"only. When unequal, the receiver copies req into ack only."
        )
    else:
        raise ValueError(f"Unknown synthetic family: {family}")
    correct = (
        f"---- MODULE {name} ----\n{extends}VARIABLES {variables}\n"
        f"vars == <<{variables}>>\nInit == {initial}\nNext == {next_action}\n"
        f"TypeOK == {invariant}\nSpec == Init /\\ [][Next]_vars\n====\n"
    )
    defect = ("header", "undefined_predicate", "incomplete_init")[defect_index % 3]
    if defect == "header":
        broken = correct.replace(f"---- MODULE {name} ----", f"MODULE {name}", 1)
        explanation = (
            "The module header is malformed. Use the TLA+ module delimiter "
            "syntax and preserve the declared module name and behavior."
        )
    elif defect == "undefined_predicate":
        broken = correct.replace(f"TypeOK == {invariant}", f"TypeOK == MissingPredicate{n}", 1)
        explanation = (
            "TypeOK refers to an undeclared identifier. Define the invariant "
            "over the declared state variables and their intended domains."
        )
    else:
        broken = correct.replace(f"Init == {initial}", "Init == TRUE", 1)
        explanation = (
            "Init leaves state variables unspecified. Give all declared "
            "variables initial values consistent with the task."
        )
    return Case(name, family, split, description, correct, broken, explanation, states, defect)


def cases() -> list[Case]:
    groups = {
        "train": [
            "modulo",
            "toggle",
            "resource",
            "saturating",
            "pool",
            "fifo",
            "stack",
            "phase",
        ],
        "valid": ["pair", "flags"],
        "test": ["mutex", "handshake"],
    }
    return [
        make_case(family, n, split, index)
        for split, families in groups.items()
        for index, family in enumerate(families)
        for n in ((2, 3, 4) if split == "train" else (2, 3))
    ]


def verify(source: str, cfg: str, name: str, jar: Path, output: Path) -> dict:
    """Direct Java checks for synthetic fixtures; no main-benchmark verifier implementation."""
    output.mkdir(parents=True, exist_ok=False)
    (output / f"{name}.tla").write_text(source)
    (output / f"{name}.cfg").write_text(cfg)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="tla-sft-") as work:
        work = Path(work)
        (work / f"{name}.tla").write_text(source)
        (work / f"{name}.cfg").write_text(cfg)
        for tool in ("sany", "tlc"):
            if tool == "tlc" and not checks["sany"]["success"]:
                break
            command = ["java", "-Xmx256m", "-cp", str(jar), "tla2sany.SANY", f"{name}.tla"]
            if tool == "tlc":
                command = [
                    "java",
                    "-Xmx256m",
                    "-XX:+UseSerialGC",
                    "-cp",
                    str(jar),
                    "tlc2.TLC",
                    "-workers",
                    "1",
                    "-metadir",
                    str(work / "states"),
                    "-config",
                    f"{name}.cfg",
                    f"{name}.tla",
                ]
            try:
                process = subprocess.run(command, cwd=work, capture_output=True, timeout=30)
                code, stdout, stderr, timeout = (
                    process.returncode,
                    process.stdout,
                    process.stderr,
                    False,
                )
            except subprocess.TimeoutExpired as exc:
                code, stdout, stderr, timeout = None, exc.stdout or b"", exc.stderr or b"", True
            (output / f"{tool}.stdout.log").write_bytes(stdout)
            (output / f"{tool}.stderr.log").write_bytes(stderr)
            log = (stdout + stderr).decode(errors="replace")
            errors = re.search(
                r"(?i)(fatal errors|parse error|lexical error|semantic errors|"
                r"\*\*\*\s*(errors?|abort))",
                log,
            )
            success = (
                code == 0
                and not timeout
                and not errors
                and f"Semantic processing of module {name}" in log
            )
            if tool == "tlc":
                success = (
                    code == 0
                    and not timeout
                    and "Model checking completed. No error has been found." in log
                )
            checks[tool] = {
                "success": bool(success),
                "exit_code": code,
                "timeout": timeout,
                "command": command,
            }
            if tool == "tlc":
                match = re.search(r"(\d+) distinct states found", log)
                checks[tool]["distinct_states"] = int(match[1]) if match else None
    dump(output / "checks.json", checks)
    return checks


def full_log(path: Path) -> str:
    return "\n".join(
        f"{tool} {stream}:\n{file.read_text()}"
        for tool in ("sany", "tlc")
        for stream in ("stdout", "stderr")
        if (file := path / f"{tool}.{stream}.log").exists()
    )


def diagnostic(log: str) -> str:
    lines = log.splitlines()
    start = next((i for i, line in enumerate(lines) if re.search(r"(?i)(error|abort)", line)), 0)
    return "\n".join(lines[start:])


def initial_messages(case: Case) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Write module {case.name}. {case.description}\n"
                f"Define Init, Next, vars, Spec and TypeOK. Use this fixed cfg:\n{case.cfg}"
            ),
        },
    ]


def records(case: Case, log: str) -> list[dict]:
    initial = initial_messages(case)
    result = [
        {
            "kind": "generation",
            "messages": initial + [{"role": "assistant", "content": case.correct}],
        }
    ]
    for kind, feedback in (("repair_full", log), ("repair_filtered", diagnostic(log))):
        result.append(
            {
                "kind": kind,
                "messages": initial
                + [
                    {"role": "assistant", "content": case.broken},
                    {
                        "role": "user",
                        "content": "Return the corrected complete module.\n" + feedback,
                    },
                    {"role": "assistant", "content": case.correct},
                ],
            }
        )
    result.append(
        {
            "kind": "explanation",
            "messages": [
                {"role": "system", "content": EXPLAIN_SYSTEM},
                {"role": "user", "content": case.broken + "\nVerification log:\n" + log},
                {"role": "assistant", "content": case.explanation},
            ],
        }
    )
    return result


def prepare(output: Path, jar: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / "generator-snapshot.py").write_bytes(Path(__file__).read_bytes())
    examples = cases()

    def check(case):
        directory = output / "evidence" / case.name
        good = verify(case.correct, case.cfg, case.name, jar, directory / "correct")
        if (
            not good.get("tlc", {}).get("success")
            or good["tlc"]["distinct_states"] != case.expected_states
        ):
            raise ValueError(f"Synthetic reference failed verification: {case.name}")
        bad = verify(case.broken, case.cfg, case.name, jar, directory / "broken")
        if bad.get("tlc", {}).get("success") or any(c["timeout"] for c in bad.values()):
            raise ValueError(f"Defect not reliably detected: {case.name}")
        # Incomplete Init must pass parsing and fail TLC; other defects must fail SANY.
        if (case.defect == "incomplete_init") != bad["sany"]["success"]:
            raise ValueError(f"Unexpected diagnostic stage: {case.name}")
        return case, records(case, full_log(directory / "broken"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        checked = list(pool.map(check, examples))
    provenance = []
    for split in ("train", "valid", "test"):
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as file:
            for case, rows in checked:
                if case.split == split:
                    for row in rows:
                        file.write(
                            json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n"
                        )
                        provenance.append(
                            {
                                "case": case.name,
                                "family": case.family,
                                "split": split,
                                "kind": row["kind"],
                                "target_sha256": hashlib.sha256(
                                    row["messages"][-1]["content"].encode()
                                ).hexdigest(),
                            }
                        )
    dump(output / "cases.json", [asdict(case) for case in examples])
    manifest = {
        "source": "authored synthetic templates; no coursework tasks or Gold files read",
        "generator_sha256": sha(output / "generator-snapshot.py"),
        "jar_sha256": sha(jar),
        "case_count": len(examples),
        "splits": {
            split: {
                "families": sorted({c.family for c in examples if c.split == split}),
                "records": sum(r["split"] == split for r in provenance),
                "sha256": sha(output / f"{split}.jsonl"),
            }
            for split in ("train", "valid", "test")
        },
        "records": provenance,
        "limitations": (
            "Parameterized synthetic cases, not independent benchmark tasks. "
            "Main evaluation tasks and close variants must be audited for overlap before "
            "coursework experiments. Tool success and state counts "
            "do not prove full semantic fidelity."
        ),
    }
    dump(output / "manifest.json", manifest)
    return manifest
