"""Benchmark CLI entry point.

Scaffold only. The real command lands in ``feature/v1-metrics-and-cli`` and will support:

    python -m aerointentbench.run_benchmark \\
      --episode data/episodes/episode_001.json \\
      --contract data/contracts/contract_001.json \\
      --policy rule_based \\
      --output results/episode_001_rule_based.json

plus a suite mode that runs a fixture set and writes aggregate metrics.

This module exists now so that the ``aerointentbench`` console script declared in
``pyproject.toml`` resolves, and so the import surface is fixed before implementation.
It deliberately does no work and reports an honest non-zero exit status.
"""

from __future__ import annotations

import sys

_NOT_IMPLEMENTED_MESSAGE = (
    "aerointentbench: the benchmark CLI is not implemented yet.\n"
    "This is the feature/v1-spec-and-scaffold skeleton; the runnable command arrives in\n"
    "feature/v1-metrics-and-cli. See docs/v1_spec.md for the planned interface."
)


def main(argv: list[str] | None = None) -> int:
    """Print the scaffold notice and return a non-zero status.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``. Accepted and
            ignored so that the signature is stable once parsing is implemented.

    Returns:
        Exit status ``2`` (usage/unavailable).
    """
    del argv
    print(_NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
