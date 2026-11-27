"""
The PoC's fixed, hand-picked partitioning scheme and the check it must pass.

Bounds are partition units as in SRS S.9: numbered 1 to N in execution order, where unit 1 is the
embedding, units 2 to N - 1 are the transformer layers and unit N is the output head. Both bounds
of a Segment are inclusive.
"""

from lake.errors import NoValidScheme
from lake.types import Scheme, Segment

DEMO_UNITS = 24
"""
N for the demo split: 22 transformer layers plus the embedding and the output head.

Illustrative only. This is TinyLlama-1.1B's expected layer count and has not been checked against
the model file yet.
"""

DEMO_SCHEME: Scheme = (
    Segment("127.0.0.1:5555", 1, 12),
    Segment("127.0.0.1:5556", 13, 24),
)
"""The demo split by hand down the middle, across two local node processes."""

POC_NODES = 2
"""The PoC splits the model across exactly two nodes."""


def check_scheme(p: Scheme, n: int) -> None:
    """
    Raise NoValidScheme unless p splits units 1 to n across exactly two distinct nodes.

    The segments must be non-empty, in execution order, and cover every unit exactly once
    (SRS S.9.2, V1 to V4). Memory (V5) is not checked in the PoC.
    """
    if len(p) != POC_NODES:
        raise NoValidScheme(f"expected {POC_NODES} segments, got {len(p)}")
    nodes = [s.node for s in p]
    if any(not node for node in nodes):
        raise NoValidScheme("a segment has an empty node")
    if len(set(nodes)) != len(nodes):
        raise NoValidScheme(f"nodes are not distinct: {nodes}")

    expected_first = 1
    for s in p:
        if s.first > s.last:
            raise NoValidScheme(f"{s.node} has an empty range {s.first} to {s.last}")
        if s.first != expected_first:
            raise NoValidScheme(f"{s.node} starts at unit {s.first}, expected {expected_first}")
        expected_first = s.last + 1
    if p[-1].last != n:
        raise NoValidScheme(f"the last segment ends at unit {p[-1].last}, expected {n}")
