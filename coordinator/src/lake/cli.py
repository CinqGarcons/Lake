"""
The lake command. Run `lake node` on every node, `lake leader "prompt"` on the leader.

Examples:
    lake node --listen 1.2.3.4:5555
    lake leader "write code" --nodes 1.2.3.4:5555,1.2.3.4:5556
"""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Primary entrypoint."""
    parser = argparse.ArgumentParser(prog="lake")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("node", help="run Core")
    leader = sub.add_parser("leader", help="run Core and the Coordinator")
    leader.add_argument("prompt")
    args = parser.parse_args(argv)

    print(f"lake {args.command}: not implemented yet")
    return 1
