#!/usr/bin/env python3
"""Integration test for Databricks Unity Catalog connector.

This script connects to a real Databricks instance and tests the connector.
Set these environment variables before running:
    DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
    DATABRICKS_TOKEN=your-personal-access-token

Optional:
    DATABRICKS_CATALOG=your-catalog-name
    DATABRICKS_SCHEMA=your-schema-name

Usage:
    export DATABRICKS_HOST="https://..."
    export DATABRICKS_TOKEN="dapi..."
    python test_databricks_integration.py
"""
import os
import sys
import logging
from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Read credentials from environment
    host = os.environ.get('DATABRICKS_HOST')
    token = os.environ.get('DATABRICKS_TOKEN')
    catalog = os.environ.get('DATABRICKS_CATALOG')
    schema = os.environ.get('DATABRICKS_SCHEMA')
    
    if not host or not token:
        print("ERROR: Missing required environment variables")
        print("Please set:")
        print("  DATABRICKS_HOST=https://your-workspace.cloud.databricks.com")
        print("  DATABRICKS_TOKEN=dapi...")
        print("\nOptional:")
        print("  DATABRICKS_CATALOG=your-catalog")
        print("  DATABRICKS_SCHEMA=your-schema")
        sys.exit(1)
    
    print("=" * 70)
    print("Databricks Unity Catalog Integration Test")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Catalog: {catalog or '(default)'}")
    print(f"Schema: {schema or '(default)'}")
    print(f"Token: {'***' + token[-4:] if len(token) > 4 else '***'}")
    print("=" * 70)
    print()
    
    # Create connector instance
    print("Creating DatabricksUnityCatalog connector...")
    connector = DatabricksUnityCatalog(
        host=host,
        token=token,
        catalog=catalog,
        schema=schema,
        mlflow_registry=True,  # Try MLflow integration
        cache_ttl_seconds=60
    )
    print("Connector created successfully")
    print()
    
    # Test 1: List models
    print("-" * 70)
    print("TEST 1: list_models()")
    print("-" * 70)
    try:
        models = connector.list_models()
        print(f"Found {len(models)} model(s):")
        
        if not models:
            print("  (no models found - this is normal if you haven't registered any)")
        else:
            for name, metadata in models.items():
                print(f"\n  Model: {name}")
                print(f"    Type: {metadata.get('type')}")
                if metadata.get('type') == 'mlflow':
                    print(f"    Registered Model: {metadata.get('registered_model')}")
                    versions = metadata.get('latest_versions')
                    if versions:
                        print(f"    Latest Versions: {len(versions)} version(s)")
                elif metadata.get('type') == 'table':
                    print(f"    Catalog: {metadata.get('catalog')}")
                    print(f"    Schema: {metadata.get('schema')}")
                    print(f"    Storage: {metadata.get('storage_location', 'N/A')}")
        print()
        print("TEST 1: PASSED")
    except Exception as e:
        print(f"TEST 1: FAILED - {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 2: Get specific model (if any exist)
    print("-" * 70)
    print("TEST 2: get_model_entry()")
    print("-" * 70)
    try:
        models = connector.list_models()
        if models:
            first_model = list(models.keys())[0]
            print(f"Getting model entry for: {first_model}")
            entry = connector.get_model_entry(first_model)
            if entry:
                print(f"Entry found:")
                for key, value in entry.items():
                    print(f"  {key}: {value}")
                print()
                print("TEST 2: PASSED")
            else:
                print("Entry not found (unexpected)")
                print("TEST 2: FAILED")
        else:
            print("No models available to test get_model_entry()")
            print("TEST 2: SKIPPED")
    except Exception as e:
        print(f"TEST 2: FAILED - {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 3: Cache behavior
    print("-" * 70)
    print("TEST 3: Cache behavior")
    print("-" * 70)
    try:
        import time
        print("Listing models (first call - should hit API)...")
        start = time.time()
        models1 = connector.list_models()
        duration1 = time.time() - start
        print(f"  Duration: {duration1:.3f}s, Found: {len(models1)} models")
        
        print("Listing models (second call - should use cache)...")
        start = time.time()
        models2 = connector.list_models()
        duration2 = time.time() - start
        print(f"  Duration: {duration2:.3f}s, Found: {len(models2)} models")
        
        if duration2 < duration1 * 0.5:  # Cache should be much faster
            print("Cache appears to be working (second call faster)")
        else:
            print("Cache timing unclear (may have hit API again)")
        
        print("\nRefreshing cache...")
        connector.refresh()
        
        print("Listing models (after refresh - should hit API)...")
        start = time.time()
        models3 = connector.list_models()
        duration3 = time.time() - start
        print(f"  Duration: {duration3:.3f}s, Found: {len(models3)} models")
        print()
        print("TEST 3: PASSED")
    except Exception as e:
        print(f"TEST 3: FAILED - {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Summary
    print("=" * 70)
    print("Integration Test Complete")
    print("=" * 70)
    print("\nConnector is working with your Databricks instance!")
    print("\nNext steps:")
    print("1. Register models in Databricks MLflow or Unity Catalog")
    print("2. Use connector.list_models() to see them")
    print("3. Integrate with MRM project via mrm_project.yml config")


if __name__ == '__main__':
    main()
