"""Compliance standard registry.

Mirrors the TestRegistry in mrm/tests/library.py:
  - decorator-based registration
  - global singleton
  - builtin auto-loading
  - dynamic module loading + entry-point discovery
"""

import importlib
import importlib.metadata
import logging
from typing import Dict, List, Optional, Type

from mrm.compliance.base import ComplianceStandard

logger = logging.getLogger(__name__)


class ComplianceRegistry:
    """Central registry for compliance standards."""

    def __init__(self):
        self._standards: Dict[str, Type[ComplianceStandard]] = {}
        # Aliases map an alternate name to a canonical standard name.
        # Used to soften historical naming inconsistencies (e.g.
        # ``eu_ai_act`` -> ``euaiact``) without breaking existing code.
        self._aliases: Dict[str, str] = {}
        # Track which aliases have been warned about so we only nag
        # once per process.
        self._aliases_warned: set = set()
        self._loaded_modules: set = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, standard_class: Type[ComplianceStandard]):
        """Register a compliance standard class (usable as decorator).

        Args:
            standard_class: ComplianceStandard subclass.

        Returns:
            The class unchanged (for use as ``@register_standard``).
        """
        if not issubclass(standard_class, ComplianceStandard):
            raise TypeError(
                f"{standard_class} must be a subclass of ComplianceStandard"
            )
        self._standards[standard_class.name] = standard_class
        logger.debug(f"Registered compliance standard: {standard_class.name}")
        return standard_class

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register ``alias`` as a deprecated alternate name for
        ``canonical``.

        Looking up ``alias`` via ``get()`` will succeed and return the
        canonical class, but emit a one-shot deprecation warning so
        users migrate.
        """
        if alias == canonical:
            return
        self._aliases[alias] = canonical

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, standard_name: str) -> Type[ComplianceStandard]:
        """Get a standard class by short name.

        Tries dynamic loading on miss (builtin → entry-points). Honors
        registered aliases with a one-shot deprecation warning.
        """
        # Direct hit -- happy path.
        if standard_name in self._standards:
            return self._standards[standard_name]

        # Alias hit -- warn once, then resolve to canonical.
        if standard_name in self._aliases:
            canonical = self._aliases[standard_name]
            if standard_name not in self._aliases_warned:
                self._aliases_warned.add(standard_name)
                logger.warning(
                    "Compliance standard alias '%s' is deprecated; "
                    "use '%s' instead.",
                    standard_name, canonical,
                )
            # Recurse to handle alias-of-alias defensively.
            return self.get(canonical)

        # Try dynamic loading then re-check (including aliases registered
        # during module import).
        self._try_load_standard(standard_name)
        if standard_name in self._standards:
            return self._standards[standard_name]
        if standard_name in self._aliases:
            return self.get(standard_name)

        available = ", ".join(self.list_standards()) or "(none loaded)"
        raise KeyError(
            f"Compliance standard '{standard_name}' not found. "
            f"Available: {available}"
        )

    def list_standards(self, jurisdiction: Optional[str] = None) -> List[str]:
        """List registered standard names, optionally filtered."""
        names = list(self._standards.keys())
        if jurisdiction:
            names = [
                n for n in names
                if self._standards[n].jurisdiction == jurisdiction
            ]
        return sorted(names)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def load_builtin_standards(self):
        """Auto-load all built-in standards shipped with MRM."""
        builtin_modules = [
            "mrm.compliance.builtin.cps230",
            "mrm.compliance.builtin.sr117",
            "mrm.compliance.builtin.sr26_2",
            "mrm.compliance.builtin.eu_ai_act",
            "mrm.compliance.builtin.osfi_e23",
        ]
        for module_name in builtin_modules:
            if module_name in self._loaded_modules:
                continue
            try:
                importlib.import_module(module_name)
                self._loaded_modules.add(module_name)
                logger.debug(f"Loaded built-in standard: {module_name}")
            except ImportError as exc:
                logger.warning(f"Could not load built-in standard {module_name}: {exc}")

    def load_custom_standards(self, standards_dir: str):
        """Import standards from a project-local directory.

        Analogous to ``TestRegistry.load_custom_tests`` and the
        ``test_paths`` / ``compliance_paths`` config key.
        """
        import sys
        from pathlib import Path

        path = Path(standards_dir)
        if not path.exists():
            return

        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))

        for py_file in path.glob("*.py"):
            module_name = py_file.stem
            full_module = f"{path.name}.{module_name}"
            if full_module in self._loaded_modules:
                continue
            try:
                importlib.import_module(full_module)
                self._loaded_modules.add(full_module)
                logger.debug(f"Loaded custom standard: {full_module}")
            except ImportError as exc:
                logger.warning(f"Could not load custom standard {full_module}: {exc}")

    def _try_load_standard(self, standard_name: str):
        """Attempt to discover an unregistered standard by name.

        Order:
          1. ``mrm.compliance.builtin.<name>``
          2. ``mrm.compliance`` entry-point group
        """
        # 1. Builtin
        full_module = f"mrm.compliance.builtin.{standard_name}"
        if full_module not in self._loaded_modules:
            try:
                importlib.import_module(full_module)
                self._loaded_modules.add(full_module)
                if standard_name in self._standards:
                    return
            except ImportError:
                pass

        # 2. Entry points
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                matches = eps.select(group="mrm.compliance")
            else:
                matches = eps.get("mrm.compliance", [])
            for ep in matches:
                if ep.name == standard_name:
                    ep.load()
                    self._loaded_modules.add(ep.value)
                    return
        except Exception:
            pass


# ------------------------------------------------------------------
# Global singleton + decorator
# ------------------------------------------------------------------

compliance_registry = ComplianceRegistry()


def register_standard(standard_class: Type[ComplianceStandard]):
    """Decorator to register a compliance standard.

    Usage::

        @register_standard
        class MyCPS230(ComplianceStandard):
            name = "cps230"
            ...
    """
    return compliance_registry.register(standard_class)


def register_alias(alias: str, canonical: str) -> None:
    """Module-level shortcut for :py:meth:`ComplianceRegistry.register_alias`."""
    compliance_registry.register_alias(alias, canonical)


# ------------------------------------------------------------------
# Compatibility aliases for two historically misnamed bundled
# standards. STRATEGY.md and the README always referenced these in
# the snake_case form (``eu_ai_act`` / ``osfi_e23``); the bundled
# class registers itself under a compact name (``euaiact`` /
# ``osfie23``). Both forms resolve; the snake_case form emits a
# one-shot deprecation warning. A future major version will drop the
# warning and the compact form may be retired entirely.
# ------------------------------------------------------------------

register_alias("eu_ai_act", "euaiact")
register_alias("osfi_e23", "osfie23")
