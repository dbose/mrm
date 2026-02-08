"""Test execution engine for MRM"""

from typing import List, Dict, Any, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path
from mrm.backends.base import BackendAdapter
from mrm.tests.library import registry
from mrm.tests.base import TestResult
from mrm.core.catalog import ModelRef, ModelSource, ModelCatalog

logger = logging.getLogger(__name__)


class TestRunner:
    """Orchestrates test execution"""
    
    def __init__(self, project_config: Dict[str, Any], backend: BackendAdapter, catalog: Optional[ModelCatalog] = None):
        """
        Initialize test runner
        
        Args:
            project_config: Project configuration
            backend: Backend adapter for storing results
            catalog: Model catalog for resolving references
        """
        self.config = project_config
        self.backend = backend
        self.catalog = catalog or ModelCatalog()
        
        # Load built-in tests
        registry.load_builtin_tests()
    
    def run_tests(
        self,
        model_configs: List[Dict],
        test_selection: Optional[List[str]] = None,
        fail_fast: bool = False,
        threads: int = 1
    ) -> Dict[str, Any]:
        """
        Run tests for multiple models
        
        Args:
            model_configs: List of model configurations
            test_selection: List of test names or suite to run (None = all)
            fail_fast: Stop on first failure
            threads: Number of parallel threads
        
        Returns:
            Dictionary of results by model
        """
        # Store all configs for ref() resolution
        self._all_model_configs = model_configs
        
        results = {}
        
        if threads == 1:
            # Sequential execution
            for model_config in model_configs:
                model_name = model_config['model']['name']
                
                try:
                    result = self._run_model_tests(model_config, test_selection)
                    results[model_name] = result
                    
                    if fail_fast and not result['all_passed']:
                        logger.info(f"Stopping due to failure in {model_name}")
                        break
                
                except Exception as e:
                    logger.error(f"Error testing {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
                    
                    if fail_fast:
                        break
        
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(
                        self._run_model_tests,
                        model_config,
                        test_selection
                    ): model_config['model']['name']
                    for model_config in model_configs
                }
                
                for future in as_completed(futures):
                    model_name = futures[future]
                    
                    try:
                        result = future.result()
                        results[model_name] = result
                        
                        if fail_fast and not result['all_passed']:
                            executor.shutdown(wait=False)
                            break
                    
                    except Exception as e:
                        logger.error(f"Error testing {model_name}: {e}")
                        results[model_name] = {'error': str(e)}
        
        return results
    
    def _run_model_tests(
        self,
        model_config: Dict,
        test_selection: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run tests for a single model
        
        Args:
            model_config: Model configuration
            test_selection: Tests to run (None = all from model config)
        
        Returns:
            Dictionary with test results
        """
        model_name = model_config['model']['name']
        logger.info(f"Running tests for model: {model_name}")
        
        # Load model
        model = self._load_model(model_config)
        
        # Load datasets
        datasets = self._load_datasets(model_config.get('datasets', {}))
        
        # Get tests to run
        tests_to_run = self._select_tests(model_config, test_selection)
        
        if not tests_to_run:
            logger.warning(f"No tests configured for {model_name}")
            return {
                'model': model_name,
                'all_passed': True,
                'test_results': {}
            }
        
        # Run tests
        test_results = {}
        all_passed = True
        
        for test_spec in tests_to_run:
            test_name = test_spec['test']
            
            try:
                # Get test class
                test_class = registry.get(test_name)
                test = test_class()
                
                # Extract config
                config = test_spec.get('config', {})
                params = test_spec.get('params', {})
                config.update(params)
                
                # Get dataset for test
                dataset_name = config.pop('dataset', 'validation')
                dataset = datasets.get(dataset_name)
                
                # Run test
                logger.info(f"  Running test: {test_name}")
                result = test.run(model=model, dataset=dataset, **config)
                
                test_results[test_name] = result
                all_passed = all_passed and result.passed
                
                status = " PASSED" if result.passed else " FAILED"
                score_str = f" (score: {result.score:.3f})" if result.score is not None else ""
                logger.info(f"    {status}{score_str}")
                
                if not result.passed and result.failure_reason:
                    logger.info(f"    Reason: {result.failure_reason}")
            
            except Exception as e:
                logger.error(f"  Error running test {test_name}: {e}")
                test_results[test_name] = TestResult(
                    passed=False,
                    failure_reason=f"Test execution error: {str(e)}"
                )
                all_passed = False
        
        # Store results in backend
        try:
            model_id = model_config['model'].get('model_id', model_name)
            self.backend.log_test_results(model_id, test_results)
        except Exception as e:
            logger.warning(f"Could not store test results: {e}")
        
        return {
            'model': model_name,
            'all_passed': all_passed,
            'test_results': test_results,
            'tests_run': len(test_results),
            'tests_passed': sum(1 for r in test_results.values() if r.passed),
            'tests_failed': sum(1 for r in test_results.values() if not r.passed)
        }
    
    def _load_model(self, model_config: Dict) -> Any:
        """
        Load model from configuration with support for multiple sources
        
        Supports:
        - Local files (pickle, joblib)
        - Python classes
        - MLflow registry
        - HuggingFace Hub (hf/org/model-name)
        - Model references (ref())
        - S3/GCS/Azure storage
        
        Args:
            model_config: Model configuration dictionary
        
        Returns:
            Loaded model object
        """
        location = model_config['model'].get('location', {})
        
        # Handle string-based references (ref(), hf/, etc.)
        if isinstance(location, str):
            from mrm.core.references import ModelReference
            ref = ModelReference(location)
            location_type = ref.type
        else:
            location_type = location.get('type', 'file')
        
        # Create ModelRef from config
        model_ref = ModelRef.from_config(location) if not isinstance(location, str) else None
        
        # Handle ref() - resolve to actual model
        if location_type == 'ref':
            ref_model_name = location.get('model') if isinstance(location, dict) else ModelReference(location).path
            logger.info(f"Resolving ref: {ref_model_name}")
            
            # Find the referenced model config
            referenced_config = None
            for cfg in getattr(self, '_all_model_configs', []):
                if cfg['model']['name'] == ref_model_name:
                    referenced_config = cfg
                    break
            
            if not referenced_config:
                raise ValueError(
                    f"Model reference '{ref_model_name}' not found. "
                    f"Ensure the model is defined in your project."
                )
            
            # Recursively load the referenced model
            return self._load_model(referenced_config)
        
        # Resolve references if ModelRef
        if model_ref and model_ref.source == ModelSource.REF:
            model_ref = self.catalog.resolve_ref(model_ref)
            logger.info(f"Resolved ref to: {model_ref}")
        
        # Load based on source type
        if location_type in ['local', 'file']:
            return self._load_local_model(model_ref or location)
        
        elif location_type == 'python_class':
            return self._load_python_class(model_ref or location)
        
        elif location_type == 'mlflow':
            return self._load_mlflow_model(model_ref or location)
        
        elif location_type in ['huggingface', 'hf']:
            return self._load_huggingface_model(model_ref or location)
        
        elif location_type == 's3':
            return self._load_s3_model(model_ref or location)
        
        elif location_type == 'catalog':
            return self._load_catalog_model(location)
        
        else:
            raise ValueError(f"Unsupported model source: {location_type}")
    
    def _load_catalog_model(self, location: Union[Dict, str]) -> Any:
        """Load model from catalog"""
        if isinstance(location, str):
            from mrm.core.references import ModelReference
            ref = ModelReference(location)
            model_path = ref.path
        else:
            model_path = location.get('path') or location.get('model_path')
        
        if not self.catalog:
            raise ValueError("Model catalog not available")
        
        logger.info(f"Loading model from catalog: {model_path}")
        return self.catalog.get(model_path)
    
    def _load_local_model(self, model_ref: Union[ModelRef, Dict, str]) -> Any:
        """Load model from local file (pickle, joblib)"""
        import pickle
        import sys

        # Extract file path
        if isinstance(model_ref, ModelRef):
            file_path = Path(model_ref.identifier)
        elif isinstance(model_ref, dict):
            file_path = Path(model_ref.get('path', ''))
        else:
            file_path = Path(str(model_ref))

        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        # Add project root (cwd) to sys.path so pickle/joblib can
        # resolve module references for custom model classes (e.g.
        # models.ccr.ccr_monte_carlo.CCRMonteCarloModel)
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Try pickle first
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.debug(f"Pickle load failed, trying joblib: {e}")

        # Try joblib
        try:
            import joblib
            return joblib.load(file_path)
        except Exception as e:
            raise ValueError(f"Could not load model from {file_path}: {e}")
    
    def _load_python_class(self, model_ref: Union[ModelRef, Dict]) -> Any:
        """Load model from Python class"""
        import importlib
        import sys
        
        # Add project root to path
        sys.path.insert(0, str(Path.cwd()))
        
        if isinstance(model_ref, ModelRef):
            module_path = model_ref.metadata.get('path', '').replace('.py', '').replace('/', '.')
            class_name = model_ref.identifier
        else:
            module_path = model_ref.get('path', '').replace('.py', '').replace('/', '.')
            class_name = model_ref.get('class', '')
        
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Instantiate or return class
        if callable(model_class):
            return model_class()
        return model_class
    
    def _load_mlflow_model(self, model_ref: Union[ModelRef, Dict]) -> Any:
        """Load model from MLflow registry"""
        if isinstance(model_ref, ModelRef):
            model_name = model_ref.identifier
            version = model_ref.metadata.get('version')
            stage = model_ref.metadata.get('stage')
        else:
            model_name = model_ref.get('model_name', '')
            version = model_ref.get('version')
            stage = model_ref.get('stage')
        
        if version:
            model_uri = f"models{model_name}/{version}"
        elif stage:
            model_uri = f"models{model_name}/{stage}"
        else:
            model_uri = f"models{model_name}/latest"
        
        try:
            import mlflow
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            raise ValueError(f"Could not load MLflow model '{model_name}': {e}")
    
    def _load_huggingface_model(self, model_ref: Union[ModelRef, Dict, str]) -> Any:
        """
        Load model from HuggingFace Hub
        
        Supports:
        - Transformers models (AutoModel, AutoTokenizer)
        - Pipeline API
        - Custom task specifications
        - String reference: hf/org/model-name or hf/org/model-name:task
        """
        # Extract repo_id and task
        if isinstance(model_ref, str):
            # hf/org/model-name or hf/org/model-name:task
            from mrm.core.references import ModelReference
            ref = ModelReference(model_ref)
            if ':' in ref.path:
                repo_id, task = ref.path.split(':', 1)
            else:
                repo_id = ref.path
                task = None
            revision = 'main'
            use_pipeline = True if task else False
        elif isinstance(model_ref, ModelRef):
            repo_id = model_ref.identifier
            revision = model_ref.metadata.get('revision', 'main')
            task = model_ref.metadata.get('task')
            use_pipeline = model_ref.metadata.get('use_pipeline', True)
        else:
            repo_id = model_ref.get('repo_id', '')
            revision = model_ref.get('revision', 'main')
            task = model_ref.get('task')
            use_pipeline = model_ref.get('use_pipeline', True)
        
        try:
            from transformers import pipeline, AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )
        
        logger.info(f"Loading HuggingFace model: {repo_id} (revision: {revision})")
        
        if use_pipeline and task:
            # Use pipeline API (simpler)
            try:
                trust_remote_code = model_ref.metadata.get('trust_remote_code', False) if isinstance(model_ref, ModelRef) else model_ref.get('trust_remote_code', False) if isinstance(model_ref, dict) else False
                return pipeline(
                    task=task,
                    model=repo_id,
                    revision=revision,
                    trust_remote_code=trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {e}, trying direct load")
        
        # Direct model loading
        try:
            trust_remote_code = model_ref.metadata.get('trust_remote_code', False) if isinstance(model_ref, ModelRef) else model_ref.get('trust_remote_code', False) if isinstance(model_ref, dict) else False
            model = AutoModel.from_pretrained(
                repo_id,
                revision=revision,
                trust_remote_code=trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
            
            # Return wrapper object
            class HFModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                
                def predict(self, X):
                    """Sklearn-like predict interface"""
                    # Tokenize inputs
                    if isinstance(X, pd.DataFrame):
                        texts = X.iloc[:, 0].tolist()
                    elif isinstance(X, list):
                        texts = X
                    else:
                        texts = [str(X)]
                    
                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                    outputs = self.model(**inputs)
                    
                    # Return logits or embeddings
                    return outputs.logits.detach().numpy() if hasattr(outputs, 'logits') else outputs.last_hidden_state.detach().numpy()
            
            return HFModelWrapper(model, tokenizer)
        
        except Exception as e:
            raise ValueError(f"Could not load HuggingFace model '{repo_id}': {e}")
    
    def _load_s3_model(self, model_ref: ModelRef) -> Any:
        """Load model from S3"""
        uri = model_ref.identifier
        
        try:
            import boto3
            import tempfile
            import pickle
            
            # Parse S3 URI
            if not uri.startswith('s3/'):
                raise ValueError(f"Invalid S3 URI: {uri}")
            
            parts = uri[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            # Download to temp file
            s3 = boto3.client('s3')
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                s3.download_file(bucket, key, tmp.name)
                
                with open(tmp.name, 'rb') as f:
                    return pickle.load(f)
        
        except Exception as e:
            raise ValueError(f"Could not load model from S3 '{uri}': {e}")
    
    def _load_datasets(self, datasets_config: Dict) -> Dict[str, Any]:
        """Load datasets from configuration"""
        datasets = {}
        
        for dataset_name, dataset_config in datasets_config.items():
            try:
                dataset_type = dataset_config.get('type')
                
                if dataset_type == 'parquet':
                    path = Path(dataset_config['path'])
                    datasets[dataset_name] = pd.read_parquet(path)
                
                elif dataset_type == 'csv':
                    path = Path(dataset_config['path'])
                    datasets[dataset_name] = pd.read_csv(path)
                
                elif dataset_type == 'pickle':
                    import pickle
                    path = Path(dataset_config['path'])
                    with open(path, 'rb') as f:
                        datasets[dataset_name] = pickle.load(f)
                
                else:
                    logger.warning(f"Unknown dataset type: {dataset_type}")
            
            except Exception as e:
                logger.error(f"Could not load dataset {dataset_name}: {e}")
        
        return datasets
    
    def _select_tests(
        self,
        model_config: Dict,
        test_selection: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Select which tests to run
        
        Args:
            model_config: Model configuration
            test_selection: Override test selection
        
        Returns:
            List of test specifications
        """
        if test_selection:
            # If test_selection provided, use it
            tests = []
            for test_name in test_selection:
                # Check if it's a test suite
                test_suites = self.config.get('test_suites', {})
                if test_name in test_suites:
                    # Expand suite
                    for suite_test in test_suites[test_name]:
                        tests.append({'test': suite_test})
                else:
                    tests.append({'test': test_name})
            return tests
        
        # Otherwise use tests from model config
        tests_config = model_config.get('tests', [])
        tests = []
        
        for test_spec in tests_config:
            if isinstance(test_spec, str):
                # Simple test name
                tests.append({'test': test_spec})
            
            elif isinstance(test_spec, dict):
                if 'test_suite' in test_spec:
                    # Test suite reference
                    suite_name = test_spec['test_suite']
                    test_suites = self.config.get('test_suites', {})
                    
                    if suite_name in test_suites:
                        for suite_test in test_suites[suite_name]:
                            tests.append({'test': suite_test})
                
                elif 'test' in test_spec:
                    # Test with config
                    tests.append(test_spec)
        
        return tests
