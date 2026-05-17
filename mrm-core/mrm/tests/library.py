"""Test registry for managing all MRM tests"""

from typing import Dict, Type, List, Optional
from mrm.tests.base import MRMTest
import importlib
import logging

logger = logging.getLogger(__name__)


class TestRegistry:
    """Central registry for all MRM tests"""
    
    def __init__(self):
        self._tests: Dict[str, Type[MRMTest]] = {}
        self._test_suites: Dict[str, List[str]] = {}
        self._loaded_modules = set()
    
    def register(self, test_class: Type[MRMTest]):
        """
        Register a test class
        
        Args:
            test_class: MRMTest subclass to register
        
        Returns:
            The test class (for use as decorator)
        """
        if not issubclass(test_class, MRMTest):
            raise TypeError(f"{test_class} must be a subclass of MRMTest")
        
        self._tests[test_class.name] = test_class
        logger.debug(f"Registered test: {test_class.name}")
        
        return test_class
    
    def get(self, test_name: str) -> Type[MRMTest]:
        """
        Get test class by name
        
        Args:
            test_name: Name of the test
        
        Returns:
            Test class
        
        Raises:
            KeyError: If test not found
        """
        if test_name not in self._tests:
            # Try to load from plugins
            self._try_load_test(test_name)
        
        if test_name not in self._tests:
            raise KeyError(f"Test '{test_name}' not found. Available tests: {self.list_tests()}")
        
        return self._tests[test_name]
    
    def list_tests(self, category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """
        List available tests
        
        Args:
            category: Filter by category (e.g., 'dataset', 'model')
            tag: Filter by tag
        
        Returns:
            List of test names
        """
        tests = list(self._tests.keys())
        
        if category:
            tests = [
                name for name in tests 
                if self._tests[name].category == category
            ]
        
        if tag:
            tests = [
                name for name in tests
                if tag in self._tests[name].tags
            ]
        
        return sorted(tests)
    
    def register_suite(self, name: str, tests: List[str]):
        """
        Register a test suite
        
        Args:
            name: Suite name
            tests: List of test names in the suite
        """
        self._test_suites[name] = tests
        logger.debug(f"Registered test suite: {name} with {len(tests)} tests")
    
    def get_suite(self, name: str) -> List[str]:
        """
        Get test suite by name
        
        Args:
            name: Suite name
        
        Returns:
            List of test names
        
        Raises:
            KeyError: If suite not found
        """
        if name not in self._test_suites:
            raise KeyError(
                f"Test suite '{name}' not found. "
                f"Available suites: {list(self._test_suites.keys())}"
            )
        
        return self._test_suites[name]
    
    def list_suites(self) -> List[str]:
        """List available test suites"""
        return list(self._test_suites.keys())
    
    def _try_load_test(self, test_name: str):
        """
        Try to dynamically load a test
        
        Args:
            test_name: Test name in format 'module.TestClass'
        """
        if '.' not in test_name:
            return
        
        module_name, _ = test_name.rsplit('.', 1)
        full_module = f"mrm.tests.builtin.{module_name}"
        
        if full_module in self._loaded_modules:
            return
        
        try:
            importlib.import_module(full_module)
            self._loaded_modules.add(full_module)
            logger.debug(f"Loaded test module: {full_module}")
        except ImportError as e:
            logger.debug(f"Could not load test module {full_module}: {e}")
    
    def load_builtin_tests(self):
        """Load all built-in tests"""
        builtin_modules = [
            'mrm.tests.builtin.tabular',
            'mrm.tests.builtin.model',
            'mrm.tests.builtin.ccr',
        ]
        
        for module_name in builtin_modules:
            try:
                importlib.import_module(module_name)
                self._loaded_modules.add(module_name)
                logger.debug(f"Loaded built-in tests from: {module_name}")
            except ImportError as e:
                logger.warning(f"Could not load built-in tests from {module_name}: {e}")
    
    def load_custom_tests(self, tests_dir: str):
        """
        Load custom tests from a directory
        
        Args:
            tests_dir: Path to custom tests directory
        """
        # TODO: Implement dynamic loading of custom tests
        pass


# Global registry instance
registry = TestRegistry()


def register_test(test_class: Type[MRMTest]):
    """
    Decorator to register a test
    
    Usage:
        @register_test
        class MyTest(MRMTest):
            ...
    """
    return registry.register(test_class)
