"""Exceptions of Core and the Coordinator, named as in the MIS."""


class NotInitialised(Exception):
    """An access program other than init was called before init."""


class UnknownDevice(Exception):
    """inviteDevices named a device that has not been discovered."""


class Unconfirmed(Exception):
    """loadModel was called while an invited device has not answered."""


class NoValidScheme(Exception):
    """No valid partitioning scheme exists for the Lake network."""


class NotLoaded(Exception):
    """serveRequest was called when no model is loaded."""


class RequestFailed(Exception):
    """A request still failed after RECOVERY_LIMIT retries."""


class Busy(Exception):
    """receiveInvite was called while the node is in a Lake network led by another device."""


class NoInvitation(Exception):
    """answerInvite was called when no invitation is waiting."""


class NotLeader(Exception):
    """A leader-only access program was called by a device other than the node's leader."""


class LoadFailed(Exception):
    """The assigned partition cannot be loaded into the node's GPU-accessible memory."""
