"""Tests for the Coordinator's PoC partition handoff."""

from unittest.mock import Mock, call

import pytest

from lake.coordinator import handoff_poc_partitions
from lake.errors import NoValidScheme
from lake.types import Segment


def test_handoff_loads_each_nodes_own_partition():
    """Send each recipient its inclusive range and the other node as both neighbours."""
    file = "models/demo.gguf"
    scheme = (Segment("node-b", 1, 3), Segment("node-a", 4, 10))
    load_partition = Mock()

    handoff_poc_partitions(file, scheme, 10, load_partition)

    assert load_partition.call_args_list == [
        call("node-b", file, 1, 3, "node-a", "node-a"),
        call("node-a", file, 4, 10, "node-b", "node-b"),
    ]


def test_handoff_rejects_invalid_scheme_before_loading():
    """A gap in the second segment prevents even the first node from receiving a call."""
    scheme = (Segment("node-b", 1, 3), Segment("node-a", 5, 10))
    load_partition = Mock()

    with pytest.raises(NoValidScheme):
        handoff_poc_partitions("models/demo.gguf", scheme, 10, load_partition)

    load_partition.assert_not_called()
