"""The Coordinator module."""

from collections.abc import Callable, Iterator

from lake.scheme import check_scheme
from lake.types import DeviceID, DevState, NetState, Report, Request, Scheme


def handoff_poc_partitions(
    file: str,
    p: Scheme,
    n: int,
    load_partition: Callable[[DeviceID, str, int, int, DeviceID, DeviceID], None],
) -> None:
    """
    Validate the full scheme, then call load_partition once per node in execution order.

    The callable takes the recipient ID followed by the arguments to Core.load_partition.
    The first node sends activations to the last, which feeds tokens back to the first,
    so each node's previous and next neighbours are the same peer.
    """
    check_scheme(p, n)
    first, last = p
    load_partition(first.node, file, first.first, first.last, last.node, last.node)
    load_partition(last.node, file, last.first, last.last, first.node, first.node)


class Coordinator:
    """The leader's view of its Lake network, and the entry point for inference requests."""

    def __init__(self):
        self.initialised = False
        self.self: DeviceID | None = None
        self.dev: dict[DeviceID, DevState] = {}
        self.rep: dict[DeviceID, Report] = {}
        self.last_seen: dict[DeviceID, float] = {}
        self.file: str | None = None
        self.p: Scheme = ()
        self.p_next: Scheme = ()
        self.q = NetState.UNLOADED
        self.active = 0

    def init(self, device: DeviceID) -> None:
        """Start as the leader device, with no model loaded."""
        raise NotImplementedError

    def invite_devices(self, s: set[DeviceID]) -> None:
        """Invite the discovered devices in s."""
        raise NotImplementedError

    def load_model(self, file: str) -> None:
        """Split file across the confirmed nodes and load it."""
        raise NotImplementedError

    def monitor_network(self) -> None:
        """Query every node's health each HEALTH_INTERVAL."""
        raise NotImplementedError

    def serve_request(self, x: Request) -> Iterator[str]:
        """Generate a response to x, yielding each token's text."""
        raise NotImplementedError

    def unload_model(self) -> None:
        """Release every node's partition."""
        raise NotImplementedError
