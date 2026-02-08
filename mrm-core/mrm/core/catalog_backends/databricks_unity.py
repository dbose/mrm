"""Databricks Unity Catalog connector for MRM

This is a lightweight, testable scaffold that resolves catalog entries
to model pointers (MLflow URIs or file paths). It supports token-based
auth and a simple listing API. Integration with real Databricks APIs
will be implemented incrementally; unit tests should mock network calls.
"""
from typing import Dict, Any, Optional
import time
import logging
import os
import json

# Optional MLflow integration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

# Optional Databricks SDK
try:
    import databricks_sdk
    from databricks_sdk import DatabricksClient
except Exception:
    databricks_sdk = None
    DatabricksClient = None

import requests

logger = logging.getLogger(__name__)


class DatabricksUnityCatalog:
    """Connector for Databricks Unity Catalog.

    Methods are intentionally small and return serializable dicts so
    they are easy to test and to feed into `ModelRef`/`ModelCatalog`.
    """

    def __init__(self, host: str, token: Optional[str] = None, catalog: Optional[str] = None, schema: Optional[str] = None, mlflow_registry: bool = True, cache_ttl_seconds: int = 300, credential_provider: Optional[str] = None):
        self.host = host.rstrip('/') if host else None
        self.token = token
        self.catalog = catalog
        self.schema = schema
        self.mlflow_registry = mlflow_registry
        self.credential_provider = credential_provider
        self.cache_ttl = cache_ttl_seconds

        self._cache: Dict[str, Any] = {}
        self._cache_time = 0

    def _auth_header(self) -> Dict[str, str]:
        if not self.token:
            # Attempt to read from environment or databricks config could be added here
            raise ValueError("Databricks token not provided; set token or configure credential_provider")
        return {"Authorization": f"Bearer {self.token}"}

    def _refresh_cache_if_needed(self):
        if time.time() - self._cache_time > self.cache_ttl:
            self._cache = {}
            self._cache_time = time.time()

    def list_models(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List models available in Unity Catalog.

        Returns mapping: name -> metadata dict
        """
        self._refresh_cache_if_needed()

        key = f"models:{catalog or self.catalog}:{schema or self.schema}"
        if key in self._cache:
            return self._cache[key]

        models: Dict[str, Dict[str, Any]] = {}

        # First, include MLflow-registered models if requested
        if self.mlflow_registry and MlflowClient is not None:
            try:
                # Configure MLflow to use Databricks tracking if host/token provided
                if self.host and self.token:
                    mlflow.set_tracking_uri("databricks")
                    os.environ['DATABRICKS_HOST'] = self.host
                    os.environ['DATABRICKS_TOKEN'] = self.token
                    logger.debug(f"MLflow tracking configured for Databricks: {self.host}")
                
                client = MlflowClient()
                registered = client.search_registered_models() or []
                logger.debug("MLflow: found %d registered models", len(registered))

                for rm in registered:
                    # Robust extraction of model name from different return shapes
                    name = None
                    try:
                        name = getattr(rm, 'name', None)
                    except Exception:
                        name = None

                    if not name:
                        try:
                            if isinstance(rm, dict):
                                name = rm.get('name')
                        except Exception:
                            name = None

                    if not name:
                        try:
                            if hasattr(rm, 'to_dict'):
                                d = rm.to_dict()
                                name = d.get('name')
                            elif hasattr(rm, 'to_json'):
                                d = json.loads(rm.to_json())
                                name = d.get('name')
                        except Exception:
                            name = None

                    if not name:
                        logger.debug("MLflow entry missing name, skipping: %r", rm)
                        continue

                    # Extract latest_versions defensively
                    latest = None
                    try:
                        latest = getattr(rm, 'latest_versions', None)
                    except Exception:
                        latest = None
                    if latest is None and isinstance(rm, dict):
                        latest = rm.get('latest_versions')

                    models[name] = {
                        'type': 'mlflow',
                        'registered_model': name,
                        'latest_versions': latest
                    }
            except Exception as e:
                logger.debug(f"MLflow client listing failed: {e}")

        # Next, try Databricks SDK if available to list Unity Catalog tables
        catalog_name = catalog or self.catalog
        schema_name = schema or self.schema

        if DatabricksClient is not None:
            try:
                client = DatabricksClient(host=self.host, token=self.token) if self.host and self.token else DatabricksClient()
                # client.unity_catalog.list_tables may be available; try both naming conventions
                tables = []
                try:
                    tables = client.unity_catalog.list_tables(catalog_name=catalog_name, schema_name=schema_name)
                except Exception:
                    try:
                        tables = client.unity_catalog.list_tables(catalog=catalog_name, schema=schema_name)
                    except Exception:
                        # some SDK versions may vary; fall back to other methods
                        tables = []

                for t in tables or []:
                    tname = getattr(t, 'name', None) or (t.get('name') if isinstance(t, dict) else None)
                    if not tname:
                        continue
                    models[tname] = {
                        'type': 'table',
                        'catalog': catalog_name,
                        'schema': schema_name,
                        'storage_location': getattr(t, 'storage_location', None) or (t.get('storage_location') if isinstance(t, dict) else None)
                    }
            except Exception as e:
                logger.debug(f"Databricks SDK listing failed: {e}")

        else:
            # Fallback to REST API via requests
            try:
                if not self.host:
                    raise ValueError("Databricks host not configured for REST API listing")

                headers = self._auth_header()
                params = {}
                if catalog_name:
                    params['catalog_name'] = catalog_name
                if schema_name:
                    params['schema_name'] = schema_name

                url = f"{self.host}/api/2.1/unity-catalog/tables"
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    body = resp.json()
                    for t in body.get('tables', []):
                        tname = t.get('name')
                        models[tname] = {
                            'type': 'table',
                            'catalog': catalog_name,
                            'schema': schema_name,
                            'storage_location': t.get('storage_location')
                        }
                else:
                    logger.debug(f"Databricks REST listing returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.debug(f"Databricks REST listing failed: {e}")

        # Cache and return
        self._cache[key] = models
        return models

    def get_model_entry(self, name: str, catalog: Optional[str] = None, schema: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a model entry by name. Returns metadata or None if not found."""
        models = self.list_models(catalog=catalog, schema=schema)
        return models.get(name)

    def register_model(self, name: str, source_uri: str, catalog: Optional[str] = None, schema: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, validation_data: Optional[Any] = None) -> Dict[str, Any]:
        """Register or update a model pointer in Unity Catalog / MLflow.

        Args:
            validation_data: Optional pandas DataFrame or numpy array for signature inference
        """
        # Validate inputs
        if not name or not source_uri:
            raise ValueError("name and source_uri are required to register a model")

        entry = {
            'name': name,
            'source': source_uri,
            'catalog': catalog or self.catalog,
            'schema': schema or self.schema,
            'metadata': metadata or {},
            'registered_at': int(time.time())
        }

        # If MLflow integration requested, try to register in MLflow Model Registry
        if self.mlflow_registry:
            if mlflow is None:
                raise ValueError("mlflow not installed; cannot register model to Databricks")
            
            try:
                # Configure MLflow to use Databricks tracking
                if self.host and self.token:
                    mlflow.set_tracking_uri("databricks")
                    # CRITICAL: Set registry URI to databricks-uc for Unity Catalog support
                    mlflow.set_registry_uri("databricks-uc")
                    os.environ['DATABRICKS_HOST'] = self.host
                    os.environ['DATABRICKS_TOKEN'] = self.token
                    logger.info(f"MLflow tracking configured for Databricks Unity Catalog: {self.host}")
                else:
                    raise ValueError("Databricks host and token required for MLflow registration")
                
                # Set or create experiment - use simple path that works in Databricks free tier
                # For Unity Catalog, we'll use the model name directly in three-level namespace
                experiment_name = "/Shared/mrm-experiments"
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        # Try to create /Shared/mrm-experiments
                        try:
                            experiment_id = mlflow.create_experiment(experiment_name)
                            logger.info(f"Created MLflow experiment: {experiment_name}")
                        except Exception:
                            # Fall back to /Workspace/Shared if /Shared doesn't work
                            experiment_name = "/Workspace/Shared/mrm-experiments"
                            experiment = mlflow.get_experiment_by_name(experiment_name)
                            if experiment is None:
                                experiment_id = mlflow.create_experiment(experiment_name)
                                logger.info(f"Created MLflow experiment: {experiment_name}")
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"Using MLflow experiment: {experiment_name}")
                except Exception as exp_error:
                    # Experiments may not be creatable in free tier, just log a warning and continue
                    logger.warning(f"Could not set experiment: {exp_error}. Will use default experiment.")
                    # Don't fail - continue with default experiment
                
                
                # Load and log the model to MLflow using sklearn pattern
                # This creates the artifact and makes it available for registration
                if os.path.exists(source_uri) and os.path.isfile(source_uri):
                    import pickle
                    import sys
                    from mlflow.models.signature import infer_signature
                    
                    # Add the project directory to sys.path to handle pickle imports
                    # This allows models pickled with local module references to be loaded
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(source_uri)))
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    
                    with mlflow.start_run():
                        # Log the pickled model as sklearn model
                        with open(source_uri, 'rb') as f:
                            model_obj = pickle.load(f)
                        
                        # Infer signature from validation data or model
                        signature = None
                        try:
                            if validation_data is not None:
                                # Use provided validation data (best option)
                                import pandas as pd
                                if isinstance(validation_data, pd.DataFrame):
                                    # Separate target if it exists
                                    X = validation_data.drop('target', axis=1) if 'target' in validation_data.columns else validation_data
                                    y_pred = model_obj.predict(X.head(1))
                                    signature = infer_signature(X.head(1), y_pred)
                                    logger.info("Signature inferred from validation data")
                                else:
                                    # Numpy array
                                    y_pred = model_obj.predict(validation_data[:1])
                                    signature = infer_signature(validation_data[:1], y_pred)
                                    logger.info("Signature inferred from validation data")
                            elif hasattr(model_obj, 'feature_names_in_'):
                                # Try to infer from model feature names
                                import numpy as np
                                import pandas as pd
                                # Create a DataFrame with proper column names
                                sample_input = pd.DataFrame(
                                    np.zeros((1, len(model_obj.feature_names_in_))),
                                    columns=model_obj.feature_names_in_
                                )
                                sample_output = model_obj.predict(sample_input)
                                signature = infer_signature(sample_input, sample_output)
                                logger.info("Signature inferred from model feature names")
                            else:
                                # Last resort: create minimal signature with generic feature names
                                import numpy as np
                                import pandas as pd
                                # Try to get n_features from the model
                                n_features = None
                                if hasattr(model_obj, 'n_features_in_'):
                                    n_features = model_obj.n_features_in_
                                elif hasattr(model_obj, 'coef_'):
                                    n_features = len(model_obj.coef_[0]) if len(model_obj.coef_.shape) > 1 else len(model_obj.coef_)
                                
                                if n_features:
                                    # Create generic feature names
                                    sample_input = pd.DataFrame(
                                        np.zeros((1, n_features)),
                                        columns=[f'feature_{i}' for i in range(n_features)]
                                    )
                                    sample_output = model_obj.predict(sample_input)
                                    signature = infer_signature(sample_input, sample_output)
                                    logger.info(f"Signature inferred with {n_features} generic features")
                                else:
                                    raise ValueError(
                                        "Cannot infer signature - model has no feature metadata. "
                                        "Please provide validation_data parameter or ensure model has feature_names_in_"
                                    )
                        except Exception as sig_error:
                            logger.error(f"Failed to infer signature: {sig_error}")
                            raise ValueError(
                                f"Unity Catalog requires a model signature. Failed to infer: {sig_error}. "
                                "Please ensure validation data is available or model has feature metadata."
                            ) from sig_error
                        
                        # Use Unity Catalog three-level namespace: catalog.schema.model_name
                        uc_model_name = f"{entry['catalog'] or 'workspace'}.{entry['schema'] or 'default'}.{entry['name']}"
                        
                        # Log model with Unity Catalog registration
                        mlflow.sklearn.log_model(
                            sk_model=model_obj,
                            artifact_path="model",
                            registered_model_name=uc_model_name,
                            signature=signature
                        )
                        
                        model_uri = mlflow.get_artifact_uri("model")
                        run_id = mlflow.active_run().info.run_id
                        
                        entry['mlflow'] = {
                            'model_uri': model_uri,
                            'registered_model_name': uc_model_name,
                            'run_id': run_id,
                            'registry_ref': '1'  # Unity Catalog uses versions
                        }
                        logger.info(f"Successfully registered model '{uc_model_name}' to Databricks Unity Catalog")
                else:
                    raise ValueError(f"Model file not found: {source_uri}")

            except Exception as e:
                logger.error(f"MLflow registration failed: {e}")
                raise RuntimeError(f"Failed to register model to Databricks: {e}") from e

        # In a real impl we'd also persist a pointer into Unity Catalog metastore
        return entry

    def refresh(self):
        """Clear cache and force next listing to refresh."""
        self._cache = {}
        self._cache_time = 0
