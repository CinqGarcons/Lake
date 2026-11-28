"""Tests for the PoC partitioning scheme check."""

import pytest

from lake.errors import NoValidScheme
from lake.scheme import DEMO_SCHEME, DEMO_UNITS, check_scheme
from lake.types import Segment

A = "127.0.0.1:5555"
B = "127.0.0.1:5556"


def test_demo_scheme_is_valid():
    """The fixed demo split passes the check."""
    check_scheme(DEMO_SCHEME, DEMO_UNITS)


def test_demo_scheme_splits_down_the_middle():
    """The demo split gives units 1 to 12 and 13 to 24 to two different nodes."""
    assert [(s.first, s.last) for s in DEMO_SCHEME] == [(1, 12), (13, 24)]
    assert DEMO_SCHEME[0].node != DEMO_SCHEME[1].node


@pytest.mark.parametrize("boundary", [1, 12, 23])
def test_any_boundary_is_valid(boundary):
    """Any split of 1 to N into two non-empty ranges passes."""
    check_scheme((Segment(A, 1, boundary), Segment(B, boundary + 1, 24)), 24)


@pytest.mark.parametrize(
    "p",
    [
        pytest.param((Segment(A, 1, 12), Segment(B, 14, 24)), id="gap"),
        pytest.param((Segment(A, 1, 12), Segment(B, 12, 24)), id="overlap"),
        pytest.param((Segment(A, 1, 12), Segment(B, 13, 23)), id="missing-end"),
        pytest.param((Segment(A, 1, 12), Segment(B, 13, 25)), id="past-end"),
        pytest.param((Segment(A, 2, 12), Segment(B, 13, 24)), id="missing-start"),
        pytest.param((Segment(A, 0, 12), Segment(B, 13, 24)), id="zero-based"),
        pytest.param((Segment(A, 13, 24), Segment(B, 1, 12)), id="out-of-order"),
        pytest.param((Segment(A, 1, 0), Segment(B, 1, 24)), id="empty-range"),
        pytest.param((Segment(A, 1, 12), Segment(A, 13, 24)), id="same-node"),
        pytest.param((Segment("", 1, 12), Segment(B, 13, 24)), id="empty-node"),
        pytest.param((Segment(A, 1, 24),), id="one-segment"),
        pytest.param(
            (Segment(A, 1, 8), Segment(B, 9, 16), Segment("127.0.0.1:5557", 17, 24)),
            id="three-segments",
        ),
        pytest.param((), id="no-segments"),
    ],
)
def test_invalid_scheme_is_rejected(p):
    """A scheme that does not split units 1 to N across exactly two nodes is rejected."""
    with pytest.raises(NoValidScheme):
        check_scheme(p, 24)
