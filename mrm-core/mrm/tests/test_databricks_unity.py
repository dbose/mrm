import os
import tempfile
import types

import pytest

from mrm.core.catalog_backends import databricks_unity


def test_register_model_calls_mlflow(monkeypatch, tmp_path):
    # Prepare a fake mlflow module with register_model and start_run
    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = {}

    def fake_register_model(uri, name):
        calls['registered'] = (uri, name)
        class Dummy:
            version = '1'
        return Dummy()

    def fake_start_run():
        return FakeRun()

    fake_mlflow = types.SimpleNamespace(register_model=fake_register_model, start_run=lambda: FakeRun(), log_artifact=lambda p: None, get_artifact_uri=lambda: 'runs:/123/path')
    monkeypatch.setattr(databricks_unity, 'mlflow', fake_mlflow)

    connector = databricks_unity.DatabricksUnityCatalog(host='https://example', token='x', mlflow_registry=True)

    # create a temp file to register
    f = tmp_path / 'm.pkl'
    f.write_text('data')

    entry = connector.register_model(name='mymodel', source_uri=str(f))
    assert 'mlflow' in entry or 'registered_at' in entry
    assert calls.get('registered') is not None


def test_list_models_with_mocked_mlflow(monkeypatch):
    # Mock MlflowClient.search_registered_models
    class FakeRM:
        def __init__(self, name):
            self.name = name
            self.latest_versions = [{'version': '1'}]

    class FakeMlClient:
        def search_registered_models(self):
            return [FakeRM('model_a'), FakeRM('model_b')]

    monkeypatch.setattr(databricks_unity, 'MlflowClient', lambda: FakeMlClient())

    connector = databricks_unity.DatabricksUnityCatalog(host='https://example', token='x', mlflow_registry=True)
    models = connector.list_models()
    assert 'model_a' in models and 'model_b' in models


def test_databricks_sdk_listing(monkeypatch):
    # Mock databricks SDK client
    class FakeTable:
        def __init__(self, name, storage_location=None):
            self.name = name
            self.storage_location = storage_location

    class FakeUnity:
        def list_tables(self, catalog_name=None, schema_name=None):
            return [FakeTable('table_x', 'dbfs:/path/x'), FakeTable('table_y', 'dbfs:/path/y')]

    class FakeClient:
        def __init__(self, host=None, token=None):
            self.unity_catalog = FakeUnity()

    monkeypatch.setattr(databricks_unity, 'DatabricksClient', FakeClient)

    connector = databricks_unity.DatabricksUnityCatalog(host='https://example', token='x', mlflow_registry=False)
    models = connector.list_models(catalog='c', schema='s')
    assert 'table_x' in models and models['table_x']['storage_location'] == 'dbfs:/path/x'
