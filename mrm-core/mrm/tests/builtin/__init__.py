"""
Built-in test library for mrm-core.

Automatically imports all test modules to ensure test registration.
"""

# Import all built-in test modules to register tests
from . import tabular
from . import model
from . import ccr

# GenAI tests (optional - requires genai dependencies)
try:
    from . import genai
    GENAI_TESTS_AVAILABLE = True
except ImportError:
    GENAI_TESTS_AVAILABLE = False

__all__ = ['tabular', 'model', 'ccr']

if GENAI_TESTS_AVAILABLE:
    __all__.append('genai')
