"""Project configuration and management for MRM"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from mrm.utils.yaml_utils import load_yaml, find_project_root, validate_project_config
from mrm.backends.base import BackendAdapter
from mrm.backends.local import LocalBackend
from mrm.core.dag import ModelDAG
from mrm.core.catalog import ModelCatalog

logger = logging.getLogger(__name__)


class Project:
    """MRM Project representation"""
    
    def __init__(
        self,
        root_path: Path,
        project_config: Dict[str, Any],
        profile_config: Dict[str, Any],
        target: str = "dev"
    ):
        """
        Initialize project
        
        Args:
            root_path: Project root directory
            project_config: Project configuration from mrm_project.yml
            profile_config: Profile configuration from profiles.yml
            target: Target environment (dev, prod, etc.)
        """
        self.root_path = root_path
        self.config = project_config
        self.profile_config = profile_config
        self.target = target
        
        self.name = project_config['name']
        self.version = project_config['version']
        
        # Initialize backend
        self.backend = self._init_backend()
        
        # Build DAG (lazily, on first use)
        self._dag: Optional[ModelDAG] = None
        
        # Build catalog (lazily, on first use)
        self._catalog: Optional[ModelCatalog] = None
    
    def _init_backend(self) -> BackendAdapter:
        """Initialize backend based on profile configuration"""
        target_config = self.profile_config.get('outputs', {}).get(self.target, {})
        backend_type = target_config.get('backend', 'local')
        
        if backend_type == 'local':
            backend_config = self.config.get('backends', {}).get('local', {})
            return LocalBackend(**backend_config)
        
        elif backend_type == 'mlflow':
            from mrm.backends.mlflow import MLflowBackend
            
            tracking_uri = target_config.get('tracking_uri')
            experiment_name = target_config.get('experiment_name', 'mrm-validation')
            
            return MLflowBackend(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name
            )
        
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    @classmethod
    def load(cls, project_path: Optional[Path] = None, profile: str = "dev") -> "Project":
        """
        Load project from directory
        
        Args:
            project_path: Path to project (defaults to current directory)
            profile: Profile to use (defaults to 'dev')
        
        Returns:
            Project instance
        """
        if project_path is None:
            project_path = find_project_root()
            
            if project_path is None:
                raise FileNotFoundError(
                    "Could not find mrm_project.yml. "
                    "Are you in an MRM project directory?"
                )
        
        # Load project config
        project_config_path = project_path / "mrm_project.yml"
        if not project_config_path.exists():
            raise FileNotFoundError(f"Project config not found: {project_config_path}")
        
        project_config = load_yaml(project_config_path)
        validate_project_config(project_config)
        
        # Load profiles config
        profiles_config_path = project_path / "profiles.yml"
        if not profiles_config_path.exists():
            # Use default profile
            profile_config = {
                'mrm': {
                    'outputs': {
                        'dev': {
                            'backend': 'local'
                        }
                    },
                    'target': 'dev'
                }
            }
        else:
            profile_config = load_yaml(profiles_config_path)
        
        # Get MRM-specific profile
        mrm_profile = profile_config.get('mrm', {})
        
        # Determine target
        if profile:
            target = profile
        else:
            target = mrm_profile.get('target', 'dev')
        
        return cls(
            root_path=project_path,
            project_config=project_config,
            profile_config=mrm_profile,
            target=target
        )
    
    def list_models(self, tier: Optional[str] = None, owner: Optional[str] = None) -> List[Dict]:
        """
        List models in project
        
        Args:
            tier: Filter by risk tier
            owner: Filter by owner
        
        Returns:
            List of model configurations
        """
        models_dir = self.root_path / "models"
        
        if not models_dir.exists():
            return []
        
        models = []
        
        for model_file in models_dir.rglob("*.yml"):
            try:
                model_config = load_yaml(model_file)
                
                if 'model' not in model_config:
                    continue
                
                # Apply filters
                if tier and model_config['model'].get('risk_tier') != tier:
                    continue
                
                if owner and model_config['model'].get('owner') != owner:
                    continue
                
                model_config['_file_path'] = str(model_file.relative_to(self.root_path))
                models.append(model_config)
            
            except Exception as e:
                logger.warning(f"Could not load model config {model_file}: {e}")
        
        return models
    
    def select_models(
        self,
        models: Optional[str] = None,
        select: Optional[str] = None,
        exclude: Optional[str] = None
    ) -> List[Dict]:
        """
        Select models using dbt-style syntax
        
        Args:
            models: Comma-separated list of model names
            select: Selection criteria (e.g., 'tier:tier_1', 'owner:credit_team')
            exclude: Models to exclude
        
        Returns:
            List of selected model configurations
        """
        all_models = self.list_models()
        
        if not models and not select:
            return all_models
        
        selected = []
        
        # Direct model selection
        if models:
            model_names = [m.strip() for m in models.split(',')]
            for model_config in all_models:
                if model_config['model']['name'] in model_names:
                    selected.append(model_config)
        
        # Selection criteria
        elif select:
            if ':' in select:
                key, value = select.split(':', 1)
                
                for model_config in all_models:
                    if key == 'tier':
                        if model_config['model'].get('risk_tier') == value:
                            selected.append(model_config)
                    elif key == 'owner':
                        if model_config['model'].get('owner') == value:
                            selected.append(model_config)
                    elif key == 'category':
                        if model_config['model'].get('category') == value:
                            selected.append(model_config)
            else:
                # Assume it's a model name
                for model_config in all_models:
                    if model_config['model']['name'] == select:
                        selected.append(model_config)
        
        # Apply exclusions
        if exclude:
            exclude_names = [m.strip() for m in exclude.split(',')]
            selected = [
                m for m in selected 
                if m['model']['name'] not in exclude_names
            ]
        
        return selected
    
    def get_test_suites(self) -> Dict[str, List[str]]:
        """Get defined test suites from project config"""
        return self.config.get('test_suites', {})
    
    def get_governance_rules(self) -> Dict[str, Any]:
        """Get governance rules from project config"""
        return self.config.get('governance', {})
    
    @property
    def dag(self) -> ModelDAG:
        """
        Get model dependency DAG (lazy loading)
        
        Returns:
            ModelDAG instance
        """
        if self._dag is None:
            self._build_dag()
        return self._dag
    
    @property
    def catalog(self) -> ModelCatalog:
        """
        Get model catalog (lazy loading)
        
        Returns:
            ModelCatalog instance
        """
        if self._catalog is None:
            self._build_catalog()
        return self._catalog
    
    def _build_dag(self):
        """Build model dependency graph from configs"""
        models = self.list_models()
        
        try:
            self._dag = ModelDAG.from_model_configs(models)
            logger.info(f"Built DAG with {len(self._dag.nodes)} nodes")
        except ValueError as e:
            logger.error(f"Error building DAG: {e}")
            # Create empty DAG as fallback
            self._dag = ModelDAG()
            for model in models:
                self._dag.add_node(model['model']['name'])
    
    def _build_catalog(self):
        """Build model catalog from configs"""
        models = self.list_models()
        # Build base catalog from project model configs
        self._catalog = ModelCatalog.from_project(models)

        # Merge external catalogs configured in project config (optional)
        try:
            catalogs_cfg = self.config.get('catalogs', {})
            for name, cfg in catalogs_cfg.items():
                ctype = cfg.get('type')
                if ctype == 'databricks_unity' or ctype == 'databricks_uc':
                    try:
                        from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog

                        host = cfg.get('host')
                        token = cfg.get('token')
                        catalog_default = cfg.get('catalog')
                        schema_default = cfg.get('schema')
                        mlflow_registry = cfg.get('mlflow_registry', True)
                        cache_ttl = cfg.get('cache_ttl_seconds', 300)

                        backend = DatabricksUnityCatalog(
                            host=host,
                            token=token,
                            catalog=catalog_default,
                            schema=schema_default,
                            mlflow_registry=mlflow_registry,
                            cache_ttl_seconds=cache_ttl
                        )

                        # Register discovered models into central ModelCatalog
                        try:
                            remote_models = backend.list_models(catalog_default, schema_default)
                            for mname, meta in remote_models.items():
                                # Build a ModelRef-like dict entry
                                model_ref = {
                                    'type': 'databricks_uc',
                                    'catalog': catalog_default,
                                    'schema': schema_default,
                                    'model': mname,
                                    **(meta or {})
                                }
                                self._catalog.register(mname, ModelRef.from_config(model_ref))
                        except Exception as e:
                            logger.warning(f"Could not list models from catalog '{name}': {e}")

                    except Exception as e:
                        logger.warning(f"Could not initialize catalog backend '{name}': {e}")
                else:
                    logger.debug(f"Skipping unknown catalog type: {ctype}")
        except Exception:
            # Ignore catalog loading errors but log
            logger.debug("No external catalogs configured or failed to load them")
        logger.info(f"Built catalog with {len(self._catalog.models)} models")
    
    def select_models_graph(
        self,
        selector: str
    ) -> List[Dict]:
        """
        Select models using graph operators (dbt-style)
        
        Syntax:
            model          - Just the model
            +model         - Model and all upstream dependencies
            model+         - Model and all downstream dependents
            +model+        - Model, upstream, and downstream
            @model         - Just the model (explicit)
            1+model        - Model and 1 level upstream
            model+2        - Model and 2 levels downstream
        
        Args:
            selector: Graph selection string
        
        Returns:
            List of selected model configurations
        """
        # Get selected node names from DAG
        selected_names = self.dag.select_nodes(selector)
        
        # Get full configs for selected models
        all_models = self.list_models()
        return [
            m for m in all_models 
            if m['model']['name'] in selected_names
        ]
