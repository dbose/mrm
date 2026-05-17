"""MRM Core - Open Source Model Risk Management Framework"""

__version__ = "0.1.0"
__author__ = "MRM Contributors"
__license__ = "Apache-2.0"

from mrm.core.project import Project
from mrm.tests.library import registry
from mrm.tests.base import MRMTest
from mrm.compliance.registry import compliance_registry
from mrm.compliance.base import ComplianceStandard

__all__ = [
    "Project",
    "registry",
    "MRMTest",
    "compliance_registry",
    "ComplianceStandard",
    "__version__",
]
