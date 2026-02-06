"""Main CLI for MRM Core"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import logging
from typing import Dict, List, Any, Optional

from mrm.core.project import Project
from mrm.engine.runner import TestRunner
from mrm.tests.library import registry

app = typer.Typer(
    name="mrm",
    help="Model Risk Management CLI - dbt for model validation",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option(None, "--template", "-t", help="Template to use"),
    backend: str = typer.Option("local", "--backend", "-b", help="Default backend")
):
    """Initialize a new MRM project"""
    from mrm.core.init import initialize_project
    
    try:
        project_path = initialize_project(project_name, template, backend)
        console.print(f" Created MRM project: [bold]{project_name}[/bold]", style="green")
        console.print(f"  Location: {project_path}")
        console.print("\nNext steps:")
        console.print(f"  cd {project_name}")
        console.print("  mrm list models")
    except Exception as e:
        console.print(f" Error initializing project: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def test(
    models: str = typer.Option(None, "--models", "-m", help="Models to test"),
    select: str = typer.Option(None, "--select", "-s", help="Selection criteria"),
    exclude: str = typer.Option(None, "--exclude", "-e", help="Models to exclude"),
    suite: str = typer.Option(None, "--suite", help="Test suite to run"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop on first failure"),
    threads: int = typer.Option(1, "--threads", "-t", help="Parallel threads"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use")
):
    """Run validation tests"""
    try:
        # Load project
        project = Project.load(profile=profile)
        
        # Select models
        model_configs = project.select_models(
            models=models,
            select=select,
            exclude=exclude
        )
        
        if not model_configs:
            console.print("No models selected", style="yellow")
            raise typer.Exit(0)
        
        console.print(f"Running tests for {len(model_configs)} model(s)...\n")
        
        # Run tests
        runner = TestRunner(project.config, project.backend, project.catalog)
        
        test_selection = None
        if suite:
            test_selection = [suite]
        
        results = runner.run_tests(
            model_configs,
            test_selection=test_selection,
            fail_fast=fail_fast,
            threads=threads
        )
        
        # Display results
        console.print()
        _display_test_results(results)
        
        # Exit with error if any tests failed
        all_passed = all(
            r.get('all_passed', False) 
            for r in results.values() 
            if 'error' not in r
        )
        
        if not all_passed:
            raise typer.Exit(1)
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error running tests: {e}", style="red")
        logger.exception(e)
        raise typer.Exit(1)


@app.command(name="list")
def list_command(
    resource: str = typer.Argument(..., help="Resource type: models, tests, suites, backends"),
    tier: str = typer.Option(None, "--tier", help="Filter by risk tier"),
    owner: str = typer.Option(None, "--owner", help="Filter by owner"),
    category: str = typer.Option(None, "--category", help="Filter by category"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use")
):
    """List project resources"""
    try:
        if resource == "models":
            project = Project.load(profile=profile)
            models = project.list_models(tier=tier, owner=owner)
            _display_models_table(models)
        
        elif resource == "tests":
            registry.load_builtin_tests()
            tests = registry.list_tests(category=category)
            _display_tests_table(tests)
        
        elif resource == "suites":
            project = Project.load(profile=profile)
            suites = project.get_test_suites()
            _display_suites_table(suites)
        
        elif resource == "backends":
            project = Project.load(profile=profile)
            backends = project.config.get('backends', {})
            _display_backends_table(backends)
        
        else:
            console.print(f"Unknown resource type: {resource}", style="red")
            console.print("Available types: models, tests, suites, backends")
            raise typer.Exit(1)
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error listing {resource}: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def debug(
    show_config: bool = typer.Option(False, "--show-config", help="Show project config"),
    show_tests: bool = typer.Option(False, "--show-tests", help="Show available tests"),
    show_dag: bool = typer.Option(False, "--show-dag", help="Show model dependency graph"),
    show_catalog: bool = typer.Option(False, "--show-catalog", help="Show model catalog"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use")
):
    """Debug project configuration"""
    try:
        project = Project.load(profile=profile)
        
        console.print(f"[bold]Project:[/bold] {project.name}")
        console.print(f"[bold]Version:[/bold] {project.version}")
        console.print(f"[bold]Root:[/bold] {project.root_path}")
        console.print(f"[bold]Profile:[/bold] {profile}")
        console.print(f"[bold]Backend:[/bold] {project.backend.__class__.__name__}")
        
        if show_config:
            console.print("\n[bold]Configuration:[/bold]")
            import json
            console.print(json.dumps(project.config, indent=2))
        
        if show_tests:
            registry.load_builtin_tests()
            console.print(f"\n[bold]Available Tests:[/bold] {len(registry.list_tests())}")
            for test_name in registry.list_tests():
                console.print(f"  - {test_name}")
        
        if show_dag:
            console.print("\n[bold]Model Dependency Graph:[/bold]")
            console.print(project.dag.visualize())
            
            console.print("\n[bold]Execution Levels:[/bold]")
            levels = project.dag.get_execution_levels()
            for i, level in enumerate(levels):
                console.print(f"  Level {i}: {level}")
        
        if show_catalog:
            console.print("\n[bold]Model Catalog:[/bold]")
            for name, ref in project.catalog.models.items():
                console.print(f"  {name}: {ref.source.value} ({ref.identifier})")
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def publish(
    model: str = typer.Argument(..., help="Model name to publish"),
    to_catalog: str = typer.Option(None, "--to", help="Target catalog (databricks, mlflow)"),
    version: str = typer.Option(None, "--version", help="Version tag"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use")
):
    """Publish model to external registry (Databricks, MLflow, etc.)"""
    try:
        project = Project.load(profile=profile)
        
        # Find the model
        model_configs = project.select_models(models=model)
        if not model_configs:
            console.print(f"Model not found: {model}", style="red")
            raise typer.Exit(1)
        
        if len(model_configs) > 1:
            console.print(f"Multiple models matched '{model}', be more specific", style="yellow")
            raise typer.Exit(1)
        
        model_config = model_configs[0]
        model_info = model_config['model']
        model_name = model_info.get('name')
        
        console.print(f"Publishing model: [bold]{model_name}[/bold]")
        
        # Get model location
        location = model_info.get('location', {})
        if isinstance(location, str):
            # Parse shorthand like "file/path"
            if location.startswith('file/'):
                model_path = location[5:]
            else:
                model_path = location
        else:
            model_path = location.get('path')
        
        if not model_path:
            console.print("Model location/path not found in config", style="red")
            raise typer.Exit(1)
        
        # Make path absolute relative to project root
        from pathlib import Path
        if not Path(model_path).is_absolute():
            model_path = str(project.root_path / model_path)
        
        if not Path(model_path).exists():
            console.print(f"Model file not found: {model_path}", style="red")
            raise typer.Exit(1)
        
        # Determine target catalog
        catalogs = project.config.get('catalogs', {})
        
        if not catalogs:
            console.print("No catalogs configured. Add a catalog section to mrm_project.yml:", style="yellow")
            console.print("""
catalogs:
  databricks:
    type: databricks_unity
    host: https://your-workspace.cloud.databricks.com
    token: ${DATABRICKS_TOKEN}
    catalog: main
    schema: models
    mlflow_registry: true
""")
            raise typer.Exit(1)
        
        # Find catalog to use
        target_cfg = None
        target_name = to_catalog
        
        if target_name:
            target_cfg = catalogs.get(target_name)
            if not target_cfg:
                console.print(f"Catalog not found: {target_name}", style="red")
                raise typer.Exit(1)
        else:
            # Use first databricks catalog
            for name, cfg in catalogs.items():
                if cfg.get('type') in ('databricks_unity', 'databricks_uc'):
                    target_cfg = cfg
                    target_name = name
                    break
        
        if not target_cfg:
            console.print("No Databricks Unity Catalog found in config", style="red")
            raise typer.Exit(1)
        
        console.print(f"Target catalog: [cyan]{target_name}[/cyan]")
        
        # Publish to Databricks
        from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog
        
        backend = DatabricksUnityCatalog(
            host=target_cfg.get('host'),
            token=target_cfg.get('token'),
            catalog=target_cfg.get('catalog'),
            schema=target_cfg.get('schema'),
            mlflow_registry=target_cfg.get('mlflow_registry', True),
            cache_ttl_seconds=target_cfg.get('cache_ttl_seconds', 300)
        )
        
        # Register
        console.print(f"Registering model artifact: {model_path}")
        
        # Load validation data for signature inference if available
        validation_data = None
        try:
            datasets = model_config.get('datasets', {})
            if 'validation' in datasets:
                val_config = datasets['validation']
                val_type = val_config.get('type', 'csv')
                val_path = val_config.get('path')
                
                if val_path and val_type == 'csv':
                    from pathlib import Path
                    import pandas as pd
                    
                    # Make path absolute relative to project root
                    if not Path(val_path).is_absolute():
                        val_path = str(project.root_path / val_path)
                    
                    if Path(val_path).exists():
                        validation_data = pd.read_csv(val_path)
                        console.print(f"Loaded validation data for signature: {val_path}")
        except Exception as e:
            console.print(f"[yellow]Could not load validation data: {e}[/yellow]")
        
        try:
            entry = backend.register_model(
                name=model_name,
                source_uri=model_path,
                validation_data=validation_data,
                metadata={
                    'version': version or model_info.get('version'),
                    'risk_tier': model_info.get('risk_tier'),
                    'owner': model_info.get('owner'),
                    'use_case': model_info.get('use_case')
                }
            )
        except Exception as e:
            console.print(f"[red] Model registration failed![/red]")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        
        console.print("[green] Model published successfully![/green]")
        console.print(f"\nRegistered as: [bold]{entry.get('name')}[/bold]")
        
        if entry.get('mlflow'):
            mlflow_info = entry['mlflow']
            console.print(f"MLflow Model URI: {mlflow_info.get('model_uri')}")
            if mlflow_info.get('registry_ref'):
                console.print(f"Registry Version: {mlflow_info.get('registry_ref')}")
        else:
            console.print("[yellow]Note: MLflow registration was not performed[/yellow]")
        
        console.print("\nNext steps:")
        console.print("  1. View in Databricks MLflow: Models > Registered Models")
        console.print("  2. Reference in other projects using catalog URIs")
        console.print(f"  3. Run: mrm catalog resolve databricks_uc://{model_name}")
        
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error publishing model: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def version():
    """Show MRM version"""
    from mrm import __version__
    console.print(f"MRM Core version: {__version__}")


def _display_test_results(results: Dict):
    """Display test results in a table"""
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Tests", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    
    for model_name, result in results.items():
        if 'error' in result:
            table.add_row(
                model_name,
                "[red]ERROR[/red]",
                "-", "-", "-"
            )
        else:
            tests_run = result.get('tests_run', 0)
            tests_passed = result.get('tests_passed', 0)
            tests_failed = result.get('tests_failed', 0)
            
            if result['all_passed']:
                status = "[green] PASSED[/green]"
            else:
                status = "[red] FAILED[/red]"
            
            table.add_row(
                model_name,
                status,
                str(tests_run),
                str(tests_passed),
                str(tests_failed)
            )
    
    console.print(table)


# ----- Catalog subcommands -----
catalog_app = typer.Typer(help="Manage external model catalogs")
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("resolve")
def catalog_resolve(
    uri: str = typer.Argument(..., help="Catalog URI, e.g. databricks_uc://catalog.schema/model_name"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use")
):
    """Resolve a catalog URI to a model entry"""
    try:
        project = Project.load(profile=profile)
        catalogs = project.config.get('catalogs', {})

        # Find first databricks_unity catalog configuration
        cfg_name = None
        for k, v in catalogs.items():
            if v.get('type') in ('databricks_unity', 'databricks_uc'):
                cfg_name = k
                cfg = v
                break

        if cfg_name is None:
            console.print("No Databricks Unity Catalog configured in project (catalogs section)", style="red")
            raise typer.Exit(1)

        from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog

        backend = DatabricksUnityCatalog(
            host=cfg.get('host'),
            token=cfg.get('token'),
            catalog=cfg.get('catalog'),
            schema=cfg.get('schema'),
            mlflow_registry=cfg.get('mlflow_registry', True),
            cache_ttl_seconds=cfg.get('cache_ttl_seconds', 300)
        )

        # parse uri like databricks_uc://catalog.schema/name or databricks_uc://catalog/schema/name
        if uri.startswith('databricks_uc://') or uri.startswith('databricks_unity://'):
            tail = uri.split('://', 1)[1]
            # support both separators
            if '/' in tail:
                catalog_schema, name = tail.rsplit('/', 1)
            elif '.' in tail:
                catalog_schema, name = tail.rsplit('.', 1)
            else:
                catalog_schema = cfg.get('catalog') or ''
                name = tail

            if '.' in catalog_schema:
                catalog, schema = catalog_schema.split('.', 1)
            elif '/' in catalog_schema:
                parts = catalog_schema.split('/')
                catalog = parts[0]
                schema = parts[1] if len(parts) > 1 else None
            else:
                catalog = catalog_schema or cfg.get('catalog')
                schema = cfg.get('schema')

            entry = backend.get_model_entry(name, catalog=catalog, schema=schema)
            if entry is None:
                console.print(f"Model not found: {name}", style="yellow")
                raise typer.Exit(1)

            import json
            console.print(json.dumps(entry, indent=2))
        else:
            console.print("Only databricks_uc:// URIs are supported by this command", style="red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f" Error resolving catalog URI: {e}", style="red")
        raise typer.Exit(1)


@catalog_app.command("add")
def catalog_add(
    name: str = typer.Option(..., '--name', '-n', help='Model name to register'),
    from_file: str = typer.Option(..., '--from-file', '-f', help='Path to model artifact file'),
    catalog: str = typer.Option(None, '--catalog', help='Catalog key from project config'),
    profile: str = typer.Option('dev', '--profile', '-p', help='Profile to use')
):
    """Register a model pointer into the configured Databricks Unity Catalog (scaffold + MLflow register if enabled)"""
    try:
        project = Project.load(profile=profile)
        catalogs = project.config.get('catalogs', {})

        if not catalogs:
            console.print("No catalogs configured in project", style="red")
            raise typer.Exit(1)

        # choose specified catalog key or first databricks_unity
        cfg = None
        if catalog:
            cfg = catalogs.get(catalog)
            if cfg is None:
                console.print(f"Catalog key not found: {catalog}", style="red")
                raise typer.Exit(1)
        else:
            for k, v in catalogs.items():
                if v.get('type') in ('databricks_unity', 'databricks_uc'):
                    cfg = v
                    break

        if cfg is None:
            console.print("No Databricks Unity Catalog configured in project", style="red")
            raise typer.Exit(1)

        from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog

        backend = DatabricksUnityCatalog(
            host=cfg.get('host'),
            token=cfg.get('token'),
            catalog=cfg.get('catalog'),
            schema=cfg.get('schema'),
            mlflow_registry=cfg.get('mlflow_registry', True),
            cache_ttl_seconds=cfg.get('cache_ttl_seconds', 300)
        )

        if not from_file or not name:
            console.print("--name and --from-file are required", style="red")
            raise typer.Exit(1)

        entry = backend.register_model(name=name, source_uri=from_file)
        import json
        console.print(json.dumps(entry, indent=2))

    except Exception as e:
        console.print(f" Error registering model: {e}", style="red")
        raise typer.Exit(1)


@catalog_app.command("refresh")
def catalog_refresh(
    catalog: str = typer.Option(None, '--catalog', help='Catalog key from project config'),
    profile: str = typer.Option('dev', '--profile', '-p', help='Profile to use')
):
    """Refresh cached catalog listings"""
    try:
        project = Project.load(profile=profile)
        catalogs = project.config.get('catalogs', {})

        if not catalogs:
            console.print("No catalogs configured in project", style="red")
            raise typer.Exit(1)

        cfg = None
        if catalog:
            cfg = catalogs.get(catalog)
        else:
            for k, v in catalogs.items():
                if v.get('type') in ('databricks_unity', 'databricks_uc'):
                    cfg = v
                    break

        if cfg is None:
            console.print("No Databricks Unity Catalog configured in project", style="red")
            raise typer.Exit(1)

        from mrm.core.catalog_backends.databricks_unity import DatabricksUnityCatalog
        backend = DatabricksUnityCatalog(
            host=cfg.get('host'),
            token=cfg.get('token'),
            catalog=cfg.get('catalog'),
            schema=cfg.get('schema'),
            mlflow_registry=cfg.get('mlflow_registry', True),
            cache_ttl_seconds=cfg.get('cache_ttl_seconds', 300)
        )

        backend.refresh()
        console.print("Catalog cache refreshed", style="green")

    except Exception as e:
        console.print(f" Error refreshing catalog: {e}", style="red")
        raise typer.Exit(1)


def _display_models_table(models: List):
    """Display models in a table"""
    if not models:
        console.print("No models found", style="yellow")
        return
    
    table = Table(title="Models", show_header=True, header_style="bold magenta")
    
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Risk Tier")
    table.add_column("Owner")
    table.add_column("File")
    
    for model_config in models:
        model = model_config['model']
        table.add_row(
            model.get('name', '-'),
            model.get('version', '-'),
            model.get('risk_tier', '-'),
            model.get('owner', '-'),
            model_config.get('_file_path', '-')
        )
    
    console.print(table)


def _display_tests_table(tests: List):
    """Display tests in a table"""
    if not tests:
        console.print("No tests found", style="yellow")
        return
    
    table = Table(title=f"Available Tests ({len(tests)})", show_header=True)
    
    table.add_column("Test Name", style="cyan")
    table.add_column("Category")
    
    for test_name in tests:
        try:
            test_class = registry.get(test_name)
            table.add_row(test_name, test_class.category)
        except:
            table.add_row(test_name, "-")
    
    console.print(table)


def _display_suites_table(suites: Dict):
    """Display test suites in a table"""
    if not suites:
        console.print("No test suites defined", style="yellow")
        return
    
    table = Table(title="Test Suites", show_header=True)
    
    table.add_column("Suite Name", style="cyan")
    table.add_column("Tests", justify="right")
    
    for suite_name, tests in suites.items():
        table.add_row(suite_name, str(len(tests)))
    
    console.print(table)


def _display_backends_table(backends: Dict):
    """Display backends in a table"""
    if not backends:
        console.print("No backends configured", style="yellow")
        return
    
    table = Table(title="Backends", show_header=True)
    
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    
    for name, config in backends.items():
        backend_type = config.get('type', 'unknown')
        table.add_row(name, backend_type)
    
    console.print(table)


if __name__ == "__main__":
    app()
