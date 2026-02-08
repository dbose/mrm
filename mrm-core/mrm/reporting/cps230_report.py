"""DEPRECATED -- Use ``mrm.compliance.report_generator`` instead.

This module is a backward-compatibility shim.  All CPS 230 logic now
lives in ``mrm.compliance.builtin.cps230.CPS230Standard``.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_cps230_report(
    model_name: str,
    model_config: Dict[str, Any],
    test_results: Dict[str, Any],
    trigger_events: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Deprecated.  Use ``generate_compliance_report('cps230', ...)``."""
    warnings.warn(
        "generate_cps230_report() is deprecated. "
        "Use mrm.compliance.report_generator.generate_compliance_report("
        "'cps230', ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mrm.compliance.report_generator import generate_compliance_report

    return generate_compliance_report(
        standard_name="cps230",
        model_name=model_name,
        model_config=model_config,
        test_results=test_results,
        trigger_events=trigger_events,
        output_path=output_path,
    )
