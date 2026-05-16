"""Tests for the unified backend / catalog resolver."""

from __future__ import annotations

import pytest

from mrm.core.backend_resolver import ResolutionError, resolve


# ---------------------------------------------------------------------------
# Precedence ladder
# ---------------------------------------------------------------------------


def test_returns_empty_dict_when_nothing_declared():
    out = resolve(
        project_cfg={},
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out == {}


def test_project_layer_wins_over_defaults(monkeypatch):
    project_cfg = {
        "backends": {"default_evidence": {"type": "local", "retention_days": 365}}
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out == {"type": "local", "retention_days": 365}


def test_profile_overrides_project():
    project_cfg = {
        "backends": {"default_evidence": {"type": "local", "retention_days": 365}}
    }
    profile_cfg = {
        "outputs": {
            "prod": {
                "backends": {
                    "default_evidence": {"type": "s3_object_lock", "bucket": "ev-prod"}
                }
            }
        }
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg=profile_cfg,
        target="prod",
        section="backends",
        role="default_evidence",
    )
    assert out["type"] == "s3_object_lock"
    assert out["bucket"] == "ev-prod"
    # Project-only keys survive the merge.
    assert out["retention_days"] == 365


def test_env_overrides_profile(monkeypatch):
    monkeypatch.setenv("MRM_BACKEND_DEFAULT_EVIDENCE_BUCKET", "env-bucket")
    project_cfg = {"backends": {"default_evidence": {"type": "local"}}}
    profile_cfg = {
        "outputs": {"dev": {"backends": {"default_evidence": {"bucket": "profile-bucket"}}}}
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg=profile_cfg,
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out["bucket"] == "env-bucket"


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MRM_BACKEND_DEFAULT_EVIDENCE_BUCKET", "env-bucket")
    out = resolve(
        project_cfg={},
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
        cli_overrides={"bucket": "cli-bucket"},
    )
    assert out["bucket"] == "cli-bucket"


def test_cli_none_does_not_clobber_chain():
    project_cfg = {"backends": {"default_evidence": {"type": "local"}}}
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
        cli_overrides={"type": None, "bucket": None},
    )
    # Type comes from project, None CLI flags are filtered out.
    assert out["type"] == "local"
    assert "bucket" not in out


# ---------------------------------------------------------------------------
# Env-var rendering
# ---------------------------------------------------------------------------


def test_jinja_env_var_renders(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.db.com")
    out = resolve(
        project_cfg={"catalogs": {"databricks": {"host": "{{ env_var('DATABRICKS_HOST') }}"}}},
        profile_cfg={},
        target="dev",
        section="catalogs",
        role="databricks",
    )
    assert out["host"] == "https://example.db.com"


def test_jinja_env_var_missing_returns_none(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    out = resolve(
        project_cfg={"catalogs": {"databricks": {"host": "{{ env_var('DATABRICKS_HOST') }}"}}},
        profile_cfg={},
        target="dev",
        section="catalogs",
        role="databricks",
    )
    assert out["host"] is None


def test_suffix_env_key_promotes_value(monkeypatch):
    monkeypatch.setenv("MY_BUCKET_NAME", "prod-bucket")
    out = resolve(
        project_cfg={"backends": {"default_evidence": {"bucket_env": "MY_BUCKET_NAME"}}},
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out == {"bucket": "prod-bucket"}


def test_suffix_env_key_missing_renders_to_none(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    out = resolve(
        project_cfg={"backends": {"default_evidence": {"bucket_env": "MISSING_KEY"}}},
        profile_cfg={},
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out == {"bucket": None}


# ---------------------------------------------------------------------------
# Legacy profile shape compatibility
# ---------------------------------------------------------------------------


def test_legacy_profile_backend_string_maps_to_default_results():
    """``outputs.dev.backend: local`` => default_results.type: local."""
    profile_cfg = {"outputs": {"dev": {"backend": "local"}}}
    out = resolve(
        project_cfg={},
        profile_cfg=profile_cfg,
        target="dev",
        section="backends",
        role="default_results",
    )
    assert out == {"type": "local"}


def test_legacy_profile_carries_sibling_keys():
    profile_cfg = {
        "outputs": {
            "dev": {
                "backend": "mlflow",
                "tracking_uri": "http://mlflow:5000",
                "experiment_name": "ccr-dev",
            }
        }
    }
    out = resolve(
        project_cfg={},
        profile_cfg=profile_cfg,
        target="dev",
        section="backends",
        role="default_results",
    )
    assert out == {
        "type": "mlflow",
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "ccr-dev",
    }


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------


def test_catalog_resolution_merges_project_and_profile(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST_PROD", "https://prod.db.com")
    monkeypatch.setenv("DATABRICKS_TOKEN_PROD", "tok123")
    project_cfg = {
        "catalogs": {"databricks": {"type": "databricks_unity", "mlflow_registry": True}}
    }
    profile_cfg = {
        "outputs": {
            "prod": {
                "catalogs": {
                    "databricks": {
                        "host": "{{ env_var('DATABRICKS_HOST_PROD') }}",
                        "token": "{{ env_var('DATABRICKS_TOKEN_PROD') }}",
                        "catalog": "workspace_prod",
                        "schema": "gold",
                        "cache_ttl_seconds": 600,
                    }
                }
            }
        }
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg=profile_cfg,
        target="prod",
        section="catalogs",
        role="databricks",
    )
    assert out["type"] == "databricks_unity"
    assert out["mlflow_registry"] is True
    assert out["host"] == "https://prod.db.com"
    assert out["token"] == "tok123"
    assert out["catalog"] == "workspace_prod"
    assert out["cache_ttl_seconds"] == 600


def test_required_role_missing_raises_resolution_error():
    with pytest.raises(ResolutionError, match="default_evidence"):
        resolve(
            project_cfg={},
            profile_cfg={},
            target="dev",
            section="backends",
            role="default_evidence",
            required=True,
        )


def test_required_message_names_every_layer():
    try:
        resolve(
            project_cfg={},
            profile_cfg={},
            target="staging",
            section="catalogs",
            role="databricks",
            required=True,
        )
    except ResolutionError as exc:
        msg = str(exc)
        assert "CLI flag overrides" in msg
        assert "Env vars" in msg
        assert "profiles.yml" in msg
        assert "mrm_project.yml" in msg
        assert "staging" in msg
        assert "databricks" in msg
    else:
        pytest.fail("ResolutionError not raised")


# ---------------------------------------------------------------------------
# Deep merge behaviour
# ---------------------------------------------------------------------------


def test_deep_merge_nested_dicts():
    project_cfg = {"backends": {"default_evidence": {"options": {"a": 1, "b": 2}}}}
    profile_cfg = {
        "outputs": {"dev": {"backends": {"default_evidence": {"options": {"b": 99, "c": 3}}}}}
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg=profile_cfg,
        target="dev",
        section="backends",
        role="default_evidence",
    )
    assert out["options"] == {"a": 1, "b": 99, "c": 3}


def test_list_values_replace_not_merge():
    project_cfg = {"backends": {"default_evidence": {"regions": ["us", "eu"]}}}
    profile_cfg = {
        "outputs": {"dev": {"backends": {"default_evidence": {"regions": ["ap"]}}}}
    }
    out = resolve(
        project_cfg=project_cfg,
        profile_cfg=profile_cfg,
        target="dev",
        section="backends",
        role="default_evidence",
    )
    # Lists replace wholesale (intuition for compliance_paths etc).
    assert out["regions"] == ["ap"]
