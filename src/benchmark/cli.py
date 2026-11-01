"""Foundation CLI: inspect contracts; experiment execution comes in a later stage."""

import argparse
import json

from benchmark import SCHEMA_VERSION, __version__
from benchmark.contracts import CheckStatus, Method


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TLA+ repair benchmark")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("schema-info", help="Print shared schema version and enum values")
    parser.parse_args(argv)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "methods": {method.value: method.name for method in Method},
                "check_statuses": [status.value for status in CheckStatus],
            },
            indent=2,
        )
    )
    return 0
