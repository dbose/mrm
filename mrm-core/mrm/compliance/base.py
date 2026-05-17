"""Base class for compliance standards.

Follows the same ABC pattern as BackendAdapter in mrm/backends/base.py.
Each implementation represents one regulatory standard (CPS 230, SR 11-7, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class ComplianceStandard(ABC):
    """Abstract base class for regulatory compliance standards.

    Subclass this to implement a new regulatory standard.  Register it
    with ``@register_standard`` so the framework discovers it automatically.

    Class-level metadata (mirrors MRMTest.name / .category pattern):

        name          -- short identifier, e.g. ``"cps230"``
        display_name  -- human-readable, e.g. ``"APRA CPS 230"``
        description   -- one-liner purpose
        jurisdiction  -- ISO country code, e.g. ``"AU"``, ``"US"``, ``"UK"``
        version       -- standard version / year
    """

    name: str = ""
    display_name: str = ""
    description: str = ""
    jurisdiction: str = ""
    version: str = ""

    def __init__(self, **config):
        """Initialise with standard-specific configuration from YAML."""
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_paragraphs(self) -> Dict[str, Dict[str, str]]:
        """Return paragraph / section definitions for this standard.

        Returns:
            Dict mapping paragraph IDs to metadata::

                {
                    "Para 8-10": {
                        "title": "Risk Identification ...",
                        "requirement": "An APRA-regulated entity ...",
                    },
                }
        """

    @abstractmethod
    def get_test_mapping(self) -> Dict[str, str]:
        """Return mapping of MRM test names to paragraph IDs.

        Returns:
            ``{"ccr.MCConvergence": "Para 30-33", ...}``
        """

    @abstractmethod
    def get_governance_checks(self) -> Dict[str, Dict[str, Any]]:
        """Return governance check definitions.

        Returns:
            Dict of ``check_name -> {description, paragraph_ref, config_key}``
        """

    @abstractmethod
    def generate_report(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        test_results: Dict[str, Any],
        trigger_events: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate a compliance report for this standard.

        Returns:
            The report content as a string (Markdown by default).
        """

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults
    # ------------------------------------------------------------------

    def get_report_header_fields(self) -> Dict[str, str]:
        """Standard-specific header fields rendered into every report."""
        return {
            "Regulatory Framework": self.display_name,
            "Jurisdiction": self.jurisdiction,
            "Standard Version": self.version,
        }

    def validate_model_config(self, model_config: Dict) -> List[str]:
        """Validate a model config against this standard's requirements.

        Returns:
            List of validation error messages (empty if valid).
        """
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, jurisdiction={self.jurisdiction})>"
