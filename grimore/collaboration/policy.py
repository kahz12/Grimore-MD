"""Library-scoped authorization kernel, deliberately independent of transports.

Callers must supply identities authenticated by a trusted server, scopes resolved
from its resource catalog and memberships loaded from its authoritative store.
These types do not authenticate client claims or resolve document ownership.
Nothing here enables collaboration on the existing HTTP/MCP/CLI surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from types import MappingProxyType


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Expected an opaque identifier, not a path or personal identifier")


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    try:
        return value.astimezone(timezone.utc)
    except OverflowError as exc:
        raise ValueError("Timestamp is outside the supported range") from exc


class Role(str, Enum):
    READER = "reader"
    REVIEWER = "reviewer"
    EDITOR = "editor"
    LIBRARY_ADMIN = "library_admin"


class Operation(str, Enum):
    READ_METADATA = "read_metadata"
    SEARCH = "search"
    READ_SOURCE = "read_source"
    ANSWER = "answer"
    READ_REVIEW = "read_review"
    EXPORT = "export"
    DECIDE_REVIEW = "decide_review"
    CHANGE_CLASSIFICATION = "change_classification"
    EDIT_DOCUMENT = "edit_document"
    INDEX = "index"
    RESTORE = "restore"
    MANAGE_ACCESS = "manage_access"
    CONFIGURE_LIBRARY = "configure_library"


_READ = frozenset({
    Operation.READ_METADATA, Operation.SEARCH, Operation.READ_SOURCE,
    Operation.ANSWER, Operation.READ_REVIEW, Operation.EXPORT,
})
_REVIEW = _READ | {Operation.DECIDE_REVIEW}
_EDIT = _REVIEW | {Operation.CHANGE_CLASSIFICATION, Operation.EDIT_DOCUMENT}
ROLE_OPERATIONS = MappingProxyType({
    Role.READER: _READ,
    Role.REVIEWER: _REVIEW,
    Role.EDITOR: _EDIT,
    Role.LIBRARY_ADMIN: frozenset(Operation),
})


@dataclass(frozen=True)
class LibraryScope:
    """Full ownership key; IDs have no meaning outside this composite scope."""

    tenant_id: str
    project_id: str
    library_id: str

    def __post_init__(self):
        for value in (self.tenant_id, self.project_id, self.library_id):
            _identifier(value)


@dataclass(frozen=True)
class Principal:
    """Server-authenticated identity, never deserialized from request arguments."""

    tenant_id: str
    subject_id: str
    active: bool = True

    def __post_init__(self):
        _identifier(self.tenant_id)
        _identifier(self.subject_id)
        if type(self.active) is not bool:
            raise ValueError("Principal active state must be boolean")


@dataclass(frozen=True)
class Membership:
    """Exactly one role per subject/library in an authoritative policy snapshot."""

    subject_id: str
    scope: LibraryScope
    role: Role
    active: bool = True
    expires_at: datetime | None = None

    def __post_init__(self):
        _identifier(self.subject_id)
        if not isinstance(self.scope, LibraryScope) or not isinstance(self.role, Role):
            raise ValueError("Membership requires a typed scope and known role")
        if type(self.active) is not bool:
            raise ValueError("Membership active state must be boolean")
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", _utc(self.expires_at))


class Reason(str, Enum):
    ALLOWED = "allowed"
    UNAUTHENTICATED = "unauthenticated"
    INACTIVE_PRINCIPAL = "inactive_principal"
    INVALID_REQUEST = "invalid_request"
    WRONG_TENANT = "wrong_tenant"
    NO_MEMBERSHIP = "no_membership"
    INACTIVE_MEMBERSHIP = "inactive_membership"
    EXPIRED_MEMBERSHIP = "expired_membership"
    OPERATION_DENIED = "operation_denied"


@dataclass(frozen=True)
class Decision:
    """Internal reason only; transport must not reveal resource existence."""

    allowed: bool
    reason: Reason


class AccessDenied(PermissionError):
    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__("Access denied")


@dataclass(frozen=True)
class AccessPolicy:
    """Immutable snapshot. Revocation requires the caller to obtain a new one.

    Reject duplicate memberships rather than unioning stale/conflicting roles.
    No global administrator, loopback exemption or shared-token shortcut.
    """

    memberships: tuple[Membership, ...] = ()

    def __post_init__(self):
        if not isinstance(self.memberships, tuple):
            raise ValueError("Policy memberships must be an immutable tuple")
        seen = set()
        for membership in self.memberships:
            if not isinstance(membership, Membership):
                raise ValueError("Policy contains an invalid membership")
            key = (membership.scope, membership.subject_id)
            if key in seen:
                raise ValueError("Duplicate subject/library memberships are not allowed")
            seen.add(key)

    def decide(
        self,
        principal: Principal | None,
        scope: LibraryScope,
        operation: Operation,
        *,
        now: datetime | None = None,
    ) -> Decision:
        """Deny unless a live, exact membership grants the explicit operation."""
        if principal is None:
            return Decision(False, Reason.UNAUTHENTICATED)
        if (not isinstance(principal, Principal) or not isinstance(scope, LibraryScope)
                or not isinstance(operation, Operation)):
            return Decision(False, Reason.INVALID_REQUEST)
        if not principal.active:
            return Decision(False, Reason.INACTIVE_PRINCIPAL)
        if principal.tenant_id != scope.tenant_id:
            return Decision(False, Reason.WRONG_TENANT)
        try:
            at = _utc(now if now is not None else datetime.now(timezone.utc))
        except ValueError:
            return Decision(False, Reason.INVALID_REQUEST)
        membership = next((member for member in self.memberships
                           if member.scope == scope and member.subject_id == principal.subject_id), None)
        if membership is None:
            return Decision(False, Reason.NO_MEMBERSHIP)
        if not membership.active:
            return Decision(False, Reason.INACTIVE_MEMBERSHIP)
        if membership.expires_at is not None and at >= membership.expires_at:
            return Decision(False, Reason.EXPIRED_MEMBERSHIP)
        if operation not in ROLE_OPERATIONS[membership.role]:
            return Decision(False, Reason.OPERATION_DENIED)
        return Decision(True, Reason.ALLOWED)

    def require(self, principal, scope, operation, *, now=None) -> None:
        decision = self.decide(principal, scope, operation, now=now)
        if not decision.allowed:
            raise AccessDenied(decision)
