"""Generic compliance report generator.

Thin entry point that looks up the requested standard from the
compliance registry and delegates to its ``generate_report()`` method.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mrm.compliance.registry import compliance_registry

logger = logging.getLogger(__name__)


def generate_compliance_report(
    standard_name: str,
    model_name: str,
    model_config: Dict[str, Any],
    test_results: Dict[str, Any],
    trigger_events: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> str:
    """Generate a compliance report for *standard_name*.

    Args:
        standard_name: Short name of the standard (e.g. ``"cps230"``).
        model_name: Name of the model being validated.
        model_config: Full parsed model YAML configuration.
        test_results: Dict mapping test names to ``TestResult`` objects.
        trigger_events: Optional list of trigger event dicts.
        output_path: If given, write the report to this file.

    Returns:
        The report content as a string.
    """
    compliance_registry.load_builtin_standards()

    standard_class = compliance_registry.get(standard_name)

    # Read standard-specific config from the model YAML
    standard_config = (
        model_config
        .get("compliance", {})
        .get("standards", {})
        .get(standard_name, {})
    )

    standard = standard_class(**standard_config)

    return standard.generate_report(
        model_name=model_name,
        model_config=model_config,
        test_results=test_results,
        trigger_events=trigger_events,
        output_path=output_path,
        **kwargs,
    )
