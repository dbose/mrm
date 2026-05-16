"""Backend / catalog config resolver.

One function does all configuration resolution in mrm-core: ``resolve``.
Every CLI command and the runner go through it, so the resolution
ladder is documented in exactly one place.

Precedence ladder (highest wins)::

    1. CLI overrides (dict passed to resolve(cli_overrides=...))
    2. Env vars (MRM_<ROLE>_<KEY> or *_env: NAME keys)
    3. profiles.yml: outputs.<target>.<section>.<role>
    4. mrm_project.yml: <section>.<role>
    5. defaults

Where ``<section>`` is ``backends`` or ``catalogs``.

The resolver:
  * deep-merges dicts at every layer
  * renders ``{{ env_var('NAME') }}`` Jinja-style references
  * renders ``<key>_env: ENV_VAR`` keys -> ``<key>: <value of env var>``
  * never raises on a missing env var (returns None, lets the caller
    decide). This is the dbt convention -- silent at parse, loud at use.

Public API
----------

  resolve(project_cfg, profile_cfg, *,
          target, section, role,
          cli_overrides=None,
          env_prefix="MRM",
          required=False) -> dict

  ResolutionError -- raised when ``required=True`` and the merged dict
  is empty. Carries a human-shaped diagnostic naming every layer.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------

class ResolutionError(LookupError):
    """Raised by ``resolve`` when a required role cannot be resolved.

    The message names every layer searched so a user can copy/paste
    the resolution into the right file.
    """


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve(
    project_cfg: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    *,
    target: str,
    section: str,
    role: str,
    cli_overrides: Optional[Mapping[str, Any]] = None,
    env_prefix: str = "MRM",
    required: bool = False,
) -> Dict[str, Any]:
    """Resolve one (section, role) config according to the precedence ladder.

    Args:
        project_cfg: the parsed ``mrm_project.yml`` dict.
        profile_cfg: the parsed ``mrm`` block from ``profiles.yml``.
        target: active target (``dev``, ``prod`` …).
        section: ``"backends"`` or ``"catalogs"``.
        role: role name within the section
              (``"default_evidence"``, ``"databricks"`` …).
        cli_overrides: dict of CLI-passed kwargs that take precedence
            over everything else. Keys with value ``None`` are skipped
            so unset Typer flags don't clobber the project/profile
            chain.
        env_prefix: prefix for the env-var override layer.
        required: when True, raise ``ResolutionError`` if every layer is
            empty.

    Returns:
        A merged config dict. May be empty when ``required=False``.
    """
    layers: list[tuple[str, Dict[str, Any]]] = [
        ("project", _section_role(project_cfg, section, role)),
        ("profile", _profile_layer(profile_cfg, target, section, role)),
        ("env",     _env_layer(env_prefix, section, role)),
        ("cli",     _filter_none(cli_overrides or {})),
    ]

    merged: Dict[str, Any] = {}
    for _, layer in layers:
        if layer:
            merged = _deep_merge(merged, layer)

    # Render env-var indirections at the very end so later layers can
    # override earlier ones cleanly.
    merged = _render_env_vars(merged)

    if required and not merged:
        raise ResolutionError(_required_message(target, section, role, env_prefix))
    return merged


# ---------------------------------------------------------------------------
# Layer extractors
# ---------------------------------------------------------------------------

def _section_role(cfg: Mapping[str, Any], section: str, role: str) -> Dict[str, Any]:
    """Pull ``cfg[section][role]`` defensively."""
    block = cfg.get(section) if cfg else None
    if not isinstance(block, Mapping):
        return {}
    value = block.get(role)
    if not isinstance(value, Mapping):
        return {}
    return deepcopy(dict(value))


def _profile_layer(
    profile_cfg: Mapping[str, Any], target: str, section: str, role: str
) -> Dict[str, Any]:
    """Pull ``outputs.<target>.<section>.<role>`` defensively.

    Also supports a flat "backend"-style profile (legacy / dbt-shaped
    minimalism) where the active target just sets ``backend: local``
    for the test-results store. That maps to
    ``backends.default_results = {type: local}``.
    """
    outputs = profile_cfg.get("outputs") if profile_cfg else None
    if not isinstance(outputs, Mapping):
        return {}
    target_cfg = outputs.get(target)
    if not isinstance(target_cfg, Mapping):
        return {}

    # Modern shape: outputs.<target>.<section>.<role>
    section_block = target_cfg.get(section)
    if isinstance(section_block, Mapping):
        role_block = section_block.get(role)
        if isinstance(role_block, Mapping):
            return deepcopy(dict(role_block))

    # Legacy shape: outputs.<target>.backend = "local" -> default_results
    if section == "backends" and role == "default_results":
        legacy_backend = target_cfg.get("backend")
        if isinstance(legacy_backend, str):
            legacy = {"type": legacy_backend}
            # Carry sibling keys that look like backend params.
            for k, v in target_cfg.items():
                if k in {"backend", section, "catalogs"}:
                    continue
                legacy.setdefault(k, v)
            return legacy

    return {}


def _env_layer(prefix: str, section: str, role: str) -> Dict[str, Any]:
    """Read env vars shaped like ``MRM_<SECTION>_<ROLE>_<KEY>``.

    Section is singularised (``BACKENDS`` -> ``BACKEND``) so a user can
    write ``MRM_BACKEND_DEFAULT_EVIDENCE_BUCKET=...`` which reads more
    naturally than ``MRM_BACKENDS_DEFAULT_EVIDENCE_BUCKET=...``.
    """
    singular = section[:-1] if section.endswith("s") else section
    env_role = role.upper().replace("-", "_")
    needle = f"{prefix}_{singular.upper()}_{env_role}_"
    out: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(needle):
            sub = key[len(needle):].lower()
            out[sub] = value
    return out


def _filter_none(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop keys whose value is ``None`` so unset CLI flags don't
    clobber the chain."""
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. ``overlay`` wins for non-dict values.

    Lists are replaced wholesale (not concatenated) — that matches user
    intuition for things like ``compliance_paths``.
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# env-var rendering
# ---------------------------------------------------------------------------

_JINJA_ENV_VAR = re.compile(r"\{\{\s*env_var\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")


def _render_env_vars(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Walk ``cfg`` and apply env-var rendering.

    Two forms are supported:

      * ``{{ env_var('NAME') }}`` — dbt-style template, anywhere a
        string value occurs. Missing env vars render to ``None``
        (silent at parse, loud at use; matches dbt's posture).

      * ``<key>_env: NAME`` — sugar for ``<key>: <env value>``. Removes
        the original ``_env`` key. Missing env vars leave the value as
        ``None``.
    """
    out: Dict[str, Any] = {}
    promoted: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(k, str) and k.endswith("_env") and isinstance(v, str):
            real_key = k[:-4]
            promoted[real_key] = os.environ.get(v)
            continue
        if isinstance(v, str):
            out[k] = _render_string(v)
        elif isinstance(v, Mapping):
            out[k] = _render_env_vars(v)
        elif isinstance(v, list):
            out[k] = [_render_string(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    # Promoted *_env values win over plain keys.
    for k, v in promoted.items():
        out[k] = v
    return out


def _render_string(s: str) -> Any:
    match = _JINJA_ENV_VAR.search(s)
    if not match:
        return s
    # If the whole string IS the env var, return the raw env value (or
    # None). Otherwise interpolate inside the string.
    if _JINJA_ENV_VAR.fullmatch(s):
        return os.environ.get(match.group(1))
    return _JINJA_ENV_VAR.sub(lambda m: os.environ.get(m.group(1), ""), s)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _required_message(target: str, section: str, role: str, env_prefix: str) -> str:
    singular = section[:-1] if section.endswith("s") else section
    env_role = role.upper().replace("-", "_")
    env_var_hint = f"{env_prefix}_{singular.upper()}_{env_role}_<KEY>"
    return (
        f"Required {section} role '{role}' could not be resolved for target '{target}'.\n"
        f"  Searched (highest to lowest precedence):\n"
        f"    1. CLI flag overrides (none, or all None)\n"
        f"    2. Env vars matching {env_var_hint}\n"
        f"    3. profiles.yml: mrm.outputs.{target}.{section}.{role}\n"
        f"    4. mrm_project.yml: {section}.{role}\n"
        f"  Add the role to at least one layer."
    )
