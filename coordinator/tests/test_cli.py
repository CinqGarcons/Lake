"""Tests for the lake command."""

import pytest

from lake.cli import main


def test_subcommand_is_required():
    """Running lake on its own is an error."""
    with pytest.raises(SystemExit):
        main([])


def test_leader_takes_a_prompt():
    """The leader subcommand parses a prompt."""
    assert main(["leader", "hi"]) == 1
