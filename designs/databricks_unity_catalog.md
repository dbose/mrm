# Databricks Unity Catalog Integration

## Status

- **Status**: ✅ **IMPLEMENTED**
- **Version**: v0.1.0
- **Implementation Date**: February 2026
- **Features**:
  - Unity Catalog three-level namespace support
  - Automatic signature inference from validation data
  - MLflow tracking and registry integration
  - dbt-style environment variable interpolation
  - CLI `mrm publish` command
  - Proper error handling and validation
- **Tested Against**: Databricks Free Edition (Community Edition)
- **Example Model**: credit_scorecard published to workspace.default.credit_scorecard

## Summary

Add an optional catalog backend for Databricks Unity Catalog (UC) so MRM projects can:
- discover existing model artifacts and metadata stored in Databricks/MLflow/UC
- use `ref()`-style references that point to Unity Catalog entries
- optionally register/save model metadata (pointers) back into Unity Catalog or MLflow Registry (when UC-enabled)

This design describes authentication options, configuration, API surface, ref resolution flow, CLI examples, caching, security considerations, and an implementation plan.

## Goals

- Provide `catalog://` or `databricks_uc://` model references that resolve to model artifacts available in Databricks Unity Catalog or MLflow with UC integration.
- Allow saving model metadata (pointers) back to the catalog, optionally registering via MLflow Registry.
- Make integration pluggable and optional; MRM must still work when Unity Catalog is not configured.
- Respect principle of least privilege for credentials; do not hard-code tokens.

## Non-goals

- Implement full Databricks workspace management or cluster provisioning. This feature only catalogs and references models.
- Replace MLflow; instead, integrate with MLflow/Databricks model registry where appropriate.

## High-level architecture

- New connector module: `mrm.core.catalog_backends.databricks_unity`
  - Exposes `DatabricksUnityCatalog` class implementing a small backend interface used by `ModelCatalog` (or registered in a factory).
  - Uses Databricks REST API or Databricks Python SDK to list catalogs/schemas/tables and interact with MLflow Registry where available.
- Integration points:
  - `ModelRef.from_config()` will accept type `databricks_uc` (or `catalog://...`) and create a ModelRef with source `HUGGINGFACE/MLFLOW/LOCAL/UC` as appropriate.
  - `TestRunner._load_catalog_model()` will ask the configured catalog backend to resolve a catalog path to a retrievable model (file path, MLflow model URI, or remote artifact pointer).

## Configuration (example)

Add a `catalogs` section to the project or profiles config:

```yaml
catalogs:
  databricks_uc:
    type: databricks_unity
    host: https://adb-123456789012345.11.azuredatabricks.net
    token: ${DATABRICKS_TOKEN}        # preferred: env var or credential provider
    catalog: hive_metastore            # optional default UC catalog
    schema: models_schema              # optional default schema
    mlflow_registry: true              # use MLflow registry when possible
    credential_provider: databricks_cli|env|azure_cli  # optional
    cache_ttl_seconds: 300
```

Model YAML using the catalog reference:

```yaml
model:
  name: credit_scorecard
  location:
    type: databricks_uc
    catalog: hive_metastore
    schema: models_schema
    model: credit_scorecard_v1
```

Or short string: `databricks_uc://hive_metastore.models_schema/credit_scorecard_v1`

## Type mapping and resolution rules

- If a catalog entry maps to a registered MLflow model (recommended), the backend returns an MLflow model URI (`models:/name/version`), and the TestRunner uses the MLflow loader.
- If the catalog entry points to a managed table or a file path in a UC-controlled location, the backend can return a file/URI pointer (e.g., `dbfs:/mnt/...`) which TestRunner passes to the local loader or appropriate handler.
- Provide clear error messages when the referenced object does not exist or user lacks permissions.

## Authentication

Support multiple auth flows (priority order):

- `token` from env var (`DATABRICKS_TOKEN`) or config.
- Databricks CLI credentials (the connector can invoke Databricks CLI credential helper or read `~/.databrickscfg`).
- Azure AD / AWS IAM based credentials when running on Databricks with workspace identity (noted as optional and advanced).

All credentials must be read from environment or local credential providers; do not store tokens in repo files.

## CLI UX and examples

- Resolve a catalog entry:

```
mrm catalog resolve databricks_uc://hive_metastore.models_schema/credit_scorecard_v1
```

- Add a model to the catalog (register metadata/pointer):

```
mrm catalog add --from-file models/credit_scorecard.pkl --catalog databricks_uc --schema models_schema --name credit_scorecard_v1
```

CLI commands are optional; core requirement is programmatic resolution via `ref()` and YAML `location.type`.

## Caching & performance

- Implement an in-memory TTL cache and optional on-disk cache for catalog listings.
- Provide `cache_ttl_seconds` config and `mrm catalog refresh` CLI command to force refresh.

## Security & permissions

- The connector will perform read/list operations requiring the caller to have `USAGE`/`SELECT` rights on Unity Catalog objects.
- For write/register flows (saving pointers), the caller needs model registry or table creation permissions; warn the user in the CLI and docs.

## Error handling

- Distinguish auth/permission errors vs not-found vs network errors. Provide actionable messages.

## Testing

- Unit tests with mocked Databricks API client.
- Integration tests are optional and require a Databricks test workspace and credentials; mark them as `integration` and skip by default.

## Implementation plan (high level)

1. Add connector scaffold `mrm/core/catalog_backends/databricks_unity.py` exposing `DatabricksUnityCatalog` class with methods:
   - `list_catalogs()`, `list_schemas(catalog)`, `list_models(catalog,schema)`, `get_model_entry(path)`, `register_model(path, metadata)`
2. Hook into `ModelCatalog.from_project()` or into catalog factory so project config `catalogs.databricks_uc` creates an instance.
3. Update `ModelRef.from_config()` and `_load_catalog_model()` to accept `databricks_uc` type.
4. Add TTL caching and credential provider integration.
5. Add unit tests and example usage in `examples/`.

Estimated effort: 1-2 days for scaffold + unit tests; 2-3 days including integration tests and CLI glue.

## Open questions for you (prompt before implementing)

1. Which authentication method do you prefer (personal token via env var, databricks CLI config, or workspace identity)?
2. Do you want the connector to register models into MLflow Registry (if available) when saving, or only store pointers/metadata in UC tables?
3. Should the CLI commands for `mrm catalog` be part of MVP or deferred to follow-up PR?

Please confirm the answers or add any constraints; I will implement after your prompt.
