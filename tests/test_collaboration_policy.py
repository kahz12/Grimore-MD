"""Pure authorization contracts; no claim of transport or database isolation."""
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from grimore.collaboration.policy import (
    AccessDenied, AccessPolicy, LibraryScope, Membership, Operation,
    Principal, Reason, Role, ROLE_OPERATIONS,
)


SCOPE = LibraryScope("team_a", "project_a", "library_a")
PRINCIPAL = Principal("team_a", "user_01")
NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)

# Independent matrix: tests do not derive their expectations from the policy.
EXPECTED = {
    Role.READER: {"read_metadata", "search", "read_source", "answer", "read_review", "export"},
    Role.REVIEWER: {"read_metadata", "search", "read_source", "answer", "read_review", "export",
                    "decide_review"},
    Role.EDITOR: {"read_metadata", "search", "read_source", "answer", "read_review", "export",
                  "decide_review", "change_classification", "edit_document"},
    Role.LIBRARY_ADMIN: {
        "read_metadata", "search", "read_source", "answer", "read_review", "export",
        "decide_review", "change_classification", "edit_document", "index", "restore",
        "manage_access", "configure_library",
    },
}


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("operation", list(Operation))
def test_explicit_role_matrix(role, operation):
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, role),))
    decision = policy.decide(PRINCIPAL, SCOPE, operation, now=NOW)
    expected = operation.value in EXPECTED[role]
    assert decision.allowed is expected
    assert decision.reason is (Reason.ALLOWED if expected else Reason.OPERATION_DENIED)


@pytest.mark.parametrize("operation", list(Operation))
def test_every_operation_is_denied_without_a_membership(operation):
    decision = AccessPolicy().decide(PRINCIPAL, SCOPE, operation, now=NOW)
    assert not decision.allowed and decision.reason is Reason.NO_MEMBERSHIP


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("other_scope,reason", [
    (replace(SCOPE, tenant_id="team_b"), Reason.WRONG_TENANT),
    (replace(SCOPE, project_id="project_b"), Reason.NO_MEMBERSHIP),
    (replace(SCOPE, library_id="library_b"), Reason.NO_MEMBERSHIP),
])
def test_admin_and_other_roles_do_not_cross_scope_boundaries(role, other_scope, reason):
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, role),))
    decision = policy.decide(PRINCIPAL, other_scope, Operation.SEARCH, now=NOW)
    assert not decision.allowed and decision.reason is reason


def test_same_subject_id_in_different_tenant_never_inherits_access():
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),))
    impostor = replace(PRINCIPAL, tenant_id="team_b")
    assert policy.decide(impostor, SCOPE, Operation.SEARCH, now=NOW).reason is Reason.WRONG_TENANT
    assert not policy.decide(impostor, replace(SCOPE, tenant_id="team_b"), Operation.SEARCH, now=NOW).allowed


def test_same_library_id_in_another_project_requires_a_separate_membership():
    second = replace(SCOPE, project_id="project_b")
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),
                           Membership(PRINCIPAL.subject_id, second, Role.READER)))
    assert policy.decide(PRINCIPAL, second, Operation.SEARCH, now=NOW).allowed
    assert not policy.decide(PRINCIPAL, second, Operation.MANAGE_ACCESS, now=NOW).allowed


def test_anonymous_and_suspended_principals_are_denied():
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),))
    assert policy.decide(None, SCOPE, Operation.SEARCH).reason is Reason.UNAUTHENTICATED
    decision = policy.decide(replace(PRINCIPAL, active=False), SCOPE, Operation.SEARCH)
    assert not decision.allowed and decision.reason is Reason.INACTIVE_PRINCIPAL


def test_different_subject_cannot_use_same_scope_membership():
    policy = AccessPolicy((Membership("user_02", SCOPE, Role.LIBRARY_ADMIN),))
    assert policy.decide(PRINCIPAL, SCOPE, Operation.SEARCH).reason is Reason.NO_MEMBERSHIP


@pytest.mark.parametrize("seconds,allowed", [(-1, False), (0, False), (1, True)])
def test_expiry_boundary(seconds, allowed):
    member = Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN,
                        expires_at=NOW + timedelta(seconds=seconds))
    decision = AccessPolicy((member,)).decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=NOW)
    assert decision.allowed is allowed
    if not allowed:
        assert decision.reason is Reason.EXPIRED_MEMBERSHIP


def test_timezone_normalization_does_not_extend_expiration():
    expiry = datetime(2026, 10, 7, 7, tzinfo=timezone(timedelta(hours=-5)))
    member = Membership(PRINCIPAL.subject_id, SCOPE, Role.READER, expires_at=expiry)
    assert member.expires_at == NOW
    assert AccessPolicy((member,)).decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=NOW).reason is Reason.EXPIRED_MEMBERSHIP


def test_inactive_membership_and_updated_snapshot_revoke_access():
    old = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),))
    revoked = AccessPolicy((replace(old.memberships[0], active=False),))
    assert old.decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=NOW).allowed
    assert revoked.decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=NOW).reason is Reason.INACTIVE_MEMBERSHIP
    # Snapshots deliberately do not promise a live distributed revocation feed.
    assert old.decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=NOW).allowed


@pytest.mark.parametrize("principal,scope,operation", [
    ({"subject_id": "user_01", "role": "library_admin"}, SCOPE, Operation.SEARCH),
    (PRINCIPAL, {"library_id": "library_a"}, Operation.SEARCH),
    (PRINCIPAL, SCOPE, "search"), (PRINCIPAL, SCOPE, "unknown_operation"),
    (PRINCIPAL, SCOPE, None), (PRINCIPAL, None, Operation.SEARCH),
])
def test_untyped_client_claims_and_unknown_operations_fail_closed(principal, scope, operation):
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),))
    decision = policy.decide(principal, scope, operation, now=NOW)
    assert not decision.allowed and decision.reason is Reason.INVALID_REQUEST


@pytest.mark.parametrize("clock", [datetime(2026, 10, 7), "2026-10-07", True])
def test_invalid_clock_fails_closed(clock):
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.LIBRARY_ADMIN),))
    assert policy.decide(PRINCIPAL, SCOPE, Operation.SEARCH, now=clock).reason is Reason.INVALID_REQUEST


@pytest.mark.parametrize("invalid", ["", "../vault", "a/b", "a\\b", "/tmp/vault", "user@example.org",
                                     "secret\n", "ñ", "x" * 65, 7, True])
def test_ids_reject_paths_controls_and_unstructured_personal_fields(invalid):
    with pytest.raises(ValueError):
        LibraryScope(invalid, "p", "l")
    with pytest.raises(ValueError):
        Principal("t", invalid)
    with pytest.raises(ValueError):
        Membership(invalid, SCOPE, Role.READER)


def test_invalid_role_flags_and_expiry_are_not_coerced():
    with pytest.raises(ValueError):
        Membership("u", SCOPE, "reader")
    with pytest.raises(ValueError):
        Membership("u", SCOPE, Role.READER, active=1)
    with pytest.raises(ValueError):
        Principal("t", "u", active="false")
    with pytest.raises(ValueError):
        Membership("u", SCOPE, Role.READER, expires_at=datetime(2026, 10, 7))
    with pytest.raises(ValueError):
        Membership("u", {}, Role.READER)


def test_conflicting_duplicate_memberships_never_union_roles():
    member = Membership(PRINCIPAL.subject_id, SCOPE, Role.READER)
    with pytest.raises(ValueError, match="Duplicate"):
        AccessPolicy((member, replace(member, role=Role.LIBRARY_ADMIN)))
    with pytest.raises(ValueError, match="Duplicate"):
        AccessPolicy((member, replace(member, active=False)))
    with pytest.raises(ValueError):
        AccessPolicy([member])
    with pytest.raises(ValueError):
        AccessPolicy(({},))


def test_policy_and_operation_matrix_are_immutable():
    member = Membership(PRINCIPAL.subject_id, SCOPE, Role.READER)
    policy = AccessPolicy((member,))
    with pytest.raises(FrozenInstanceError):
        member.role = Role.LIBRARY_ADMIN
    with pytest.raises(FrozenInstanceError):
        policy.memberships = ()
    with pytest.raises(TypeError):
        ROLE_OPERATIONS[Role.READER] = frozenset(Operation)
    assert isinstance(ROLE_OPERATIONS[Role.READER], frozenset)


def test_require_denies_before_any_caller_work_and_uses_generic_error():
    operations = []
    with pytest.raises(AccessDenied) as error:
        AccessPolicy().require(PRINCIPAL, SCOPE, Operation.READ_SOURCE, now=NOW)
        operations.append("read-private-document")
    assert operations == []
    assert str(error.value) == "Access denied"
    assert "library_a" not in str(error.value)
    assert error.value.decision.reason is Reason.NO_MEMBERSHIP
    policy = AccessPolicy((Membership(PRINCIPAL.subject_id, SCOPE, Role.READER),))
    assert policy.require(PRINCIPAL, SCOPE, Operation.READ_SOURCE, now=NOW) is None
