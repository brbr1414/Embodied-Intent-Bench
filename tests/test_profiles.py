"""Configuration profiles and the public view a policy may be shown."""

from __future__ import annotations

from pathlib import Path

import pytest

from aerointentbench.schemas import (
    ProfileCatalog,
    PublicProfileView,
    QualityTier,
    SchemaValidationError,
    check_catalog_is_profiled,
    load_profile_catalog,
)


def test_profile_fixture_loads(profiles: ProfileCatalog) -> None:
    assert profiles.platform_id == "UAV_PLATFORM_001"
    assert set(profiles) == {"CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG", "CFG_REMOTE_STRONG"}

    light = profiles.get("CFG_LOCAL_LIGHT")
    assert light.compute_latency_ms == 100.0
    assert light.onboard_energy_j == 3.0
    assert light.communication_mb == 0.0
    assert light.quality_tier is QualityTier.LOW

    remote = profiles.get("CFG_REMOTE_STRONG")
    assert remote.upload_mb == 1.5
    assert remote.download_mb == 0.05
    assert remote.communication_mb == 1.55
    assert remote.quality_tier is QualityTier.HIGH


def test_unprofiled_configuration_names_what_is_available(profiles: ProfileCatalog) -> None:
    with pytest.raises(SchemaValidationError, match="no profile for configuration"):
        profiles.get("CFG_NOT_PROFILED")


def test_quality_tiers_are_ordinal() -> None:
    """Policies compare tiers; they must not have to hardcode the names to do it."""
    assert QualityTier.LOW.rank < QualityTier.MEDIUM.rank < QualityTier.HIGH.rank
    assert max(QualityTier, key=lambda tier: tier.rank) is QualityTier.HIGH


def test_remote_is_not_a_dominated_option(profiles: ProfileCatalog) -> None:
    """The remote configuration must buy something, or no policy would ever select it.

    It costs bandwidth and fails when the link drops, so it has to win on quality --
    the server hosts a model the vehicle cannot run. If this ever inverts, the
    communication budget stops being a trade-off and becomes a tax.
    """
    remote = profiles.get("CFG_REMOTE_STRONG")
    best_local = max(
        (profiles.get(config_id) for config_id in ("CFG_LOCAL_LIGHT", "CFG_LOCAL_STRONG")),
        key=lambda profile: profile.quality_tier.rank,
    )
    assert remote.quality_tier.rank > best_local.quality_tier.rank


def test_profiles_are_ordered_by_cost_and_quality_together(profiles: ProfileCatalog) -> None:
    """A higher tier must cost more of something, or the choice would be free."""
    light = profiles.get("CFG_LOCAL_LIGHT")
    strong = profiles.get("CFG_LOCAL_STRONG")
    assert strong.quality_tier.rank > light.quality_tier.rank
    assert strong.compute_latency_ms > light.compute_latency_ms
    assert strong.onboard_energy_j > light.onboard_energy_j


# --- validation ---------------------------------------------------------------------


def _payload(**profile_overrides: object) -> dict[str, object]:
    profile = {
        "compute_latency_ms": 100.0,
        "onboard_energy_j": 3.0,
        "upload_mb": 0.0,
        "download_mb": 0.0,
        "quality_tier": "low",
    }
    profile.update(profile_overrides)
    return {
        "schema_version": "1.0",
        "platform_id": "UAV_PLATFORM_001",
        "profiles": {"CFG_A": profile},
    }


def test_profile_records_are_validated_despite_open_ended_keys(write_json) -> None:
    """The key set cannot be declared, but each record still is."""
    with pytest.raises(SchemaValidationError, match=r"unknown field\(s\) \['latency_ms'\]"):
        load_profile_catalog(write_json(_payload(latency_ms=100.0)))


def test_profile_error_message_names_the_offending_configuration(write_json) -> None:
    with pytest.raises(SchemaValidationError, match=r"profiles\['CFG_A'\]"):
        load_profile_catalog(write_json(_payload(quality_tier="excellent")))


@pytest.mark.parametrize("field", ["compute_latency_ms", "onboard_energy_j", "upload_mb"])
def test_profile_costs_may_not_be_negative(write_json, field: str) -> None:
    with pytest.raises(SchemaValidationError, match="must be >= 0.0"):
        load_profile_catalog(write_json(_payload(**{field: -1.0})))


def test_empty_profile_set_is_rejected(write_json) -> None:
    payload = _payload()
    payload["profiles"] = {}
    with pytest.raises(SchemaValidationError, match="must not be empty"):
        load_profile_catalog(write_json(payload))


def test_catalog_coverage_is_checked_against_the_episode(profiles: ProfileCatalog) -> None:
    """A missing profile should surface up front, not on whichever step first selects it."""
    check_catalog_is_profiled(
        profiles, config_ids=("CFG_LOCAL_LIGHT", "CFG_REMOTE_STRONG"), platform_id="UAV_PLATFORM_001"
    )
    with pytest.raises(SchemaValidationError, match="no profile for configuration"):
        check_catalog_is_profiled(
            profiles, config_ids=("CFG_MISSING",), platform_id="UAV_PLATFORM_001"
        )
    with pytest.raises(SchemaValidationError, match="profile catalog is for platform"):
        check_catalog_is_profiled(profiles, config_ids=(), platform_id="OTHER_PLATFORM")


# --- the public view ------------------------------------------------------------------


def test_public_view_discloses_expectations_not_exact_costs(profiles: ProfileCatalog) -> None:
    view = profiles.public_view()
    public = view.get("CFG_REMOTE_STRONG")

    assert public is not None
    assert public.expected_latency_ms == 100.0
    assert public.expected_upload_mb == 1.5
    assert public.quality_tier is QualityTier.HIGH
    assert not hasattr(public, "onboard_energy_j"), (
        "the public view must not disclose the exact energy the executor will charge"
    )


def test_public_latency_is_compute_only_so_the_network_stays_unknown(
    profiles: ProfileCatalog,
) -> None:
    """A policy estimates remote latency from the bandwidth it observes, not from foresight."""
    public = profiles.public_view().get("CFG_REMOTE_STRONG")
    assert public is not None
    assert public.expected_latency_ms == profiles.get("CFG_REMOTE_STRONG").compute_latency_ms


def test_public_view_covers_every_profiled_configuration(profiles: ProfileCatalog) -> None:
    view = profiles.public_view()
    assert len(view) == len(profiles)
    assert all(config_id in view for config_id in profiles)
    assert view.is_available


def test_hidden_view_answers_none_rather_than_raising() -> None:
    """"I was not told" is an ordinary situation for a policy, not an error."""
    view = PublicProfileView.hidden()
    assert not view.is_available
    assert len(view) == 0
    assert view.get("CFG_LOCAL_LIGHT") is None
    assert "CFG_LOCAL_LIGHT" not in view


def test_visibility_is_a_run_level_choice(profiles: ProfileCatalog, data_dir: Path) -> None:
    """Both modes are constructible from the same catalog, so it is a setting, not a build."""
    del data_dir
    assert profiles.public_view().is_available
    assert not PublicProfileView.hidden().is_available
