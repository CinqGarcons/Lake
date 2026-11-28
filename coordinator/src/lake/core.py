"""The Core module."""

from lake.types import DeviceID, MemberState, Report


class Core:
    """A node's membership in a Lake network, its partition, and its health."""

    def __init__(self):
        self.initialised = False
        self.self: DeviceID | None = None
        self.s = MemberState.IDLE
        self.leader: DeviceID | None = None
        self.file: str | None = None
        self.a: int | None = None
        self.b: int | None = None
        self.prev: DeviceID | None = None
        self.next: DeviceID | None = None
        self.last_query: float | None = None

    def init(self, device: DeviceID) -> None:
        """Start as a node with no Lake network."""
        raise NotImplementedError

    def answer_scan(self, from_: DeviceID) -> DeviceID:
        """Answer a discovery scan from device from_."""
        raise NotImplementedError

    def receive_invite(self, from_: DeviceID) -> None:
        """Record an invitation from the leader from_."""
        raise NotImplementedError

    def answer_invite(self, ok: bool) -> None:
        """Accept or decline the waiting invitation."""
        raise NotImplementedError

    def load_partition(self, file: str, a: int, b: int, prev: DeviceID, next_: DeviceID) -> None:
        """Load units a to b of file, with neighbours prev and next_."""
        raise NotImplementedError

    def release_partition(self) -> None:
        """Free the partition."""
        raise NotImplementedError

    def report_health(self) -> Report:
        """Return the node's free memory and token throughput."""
        raise NotImplementedError

    def check_leader(self) -> None:
        """Shut down if the leader has not queried this node for LEADER_TIMEOUT."""
        raise NotImplementedError
