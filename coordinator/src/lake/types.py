"""Types shared by Core and the Coordinator, as defined in the MIS."""

from dataclasses import dataclass
from enum import StrEnum

type DeviceID = str


@dataclass(frozen=True)
class Report:
    """Free GPU-accessible memory in bytes, and token throughput in tokens per second."""

    mem: int
    rate: float


class MemberState(StrEnum):
    """Whether a node has no Lake network, an invitation, a network, or a partition."""

    IDLE = "idle"
    INVITED = "invited"
    MEMBER = "member"
    LOADED = "loaded"


class DevState(StrEnum):
    """How far a device has got in joining the leader's Lake network."""

    DISCOVERED = "discovered"
    INVITED = "invited"
    DECLINED = "declined"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class NetState(StrEnum):
    """The states of the allocation state machine in SRS S.9.3."""

    UNLOADED = "U"
    ALLOCATING = "A"
    IDLE = "I"
    GENERATING = "G"
    RECOVERING = "R"
    INSUFFICIENT = "X"


@dataclass(frozen=True)
class Segment:
    """A node and the first and last layers it holds."""

    node: DeviceID
    first: int
    last: int


type Scheme = tuple[Segment, ...]


@dataclass(frozen=True)
class Request:
    """A chat request: the conversation so far, the most tokens to generate, and the temperature."""

    messages: tuple[str, ...]
    max_tokens: int
    temperature: float
