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


def _display_crosswalk_table(items: List[Dict], from_std: Optional[str], to_std: Optional[str], show_all: bool):
    """Display crosswalk in a rich table format"""
    
    # Build title
    if show_all:
        title = "Cross-Standard Compliance Crosswalk (All Mappings)"
    elif from_std and to_std:
        title = f"Crosswalk: {from_std.upper()} → {to_std.upper()}"
    elif from_std:
        title = f"Crosswalk from {from_std.upper()}"
    elif to_std:
        title = f"Crosswalk to {to_std.upper()}"
    else:
        title = "Compliance Crosswalk"
    
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Concept", style="cyan", width=30)
    
    if show_all or not (from_std and to_std):
        # Show all four standards
        table.add_column("CPS 230 (AU)", width=15)
        table.add_column("SR 11-7 (US)", width=15)
        table.add_column("EU AI Act (EU)", width=15)
        table.add_column("OSFI E-23 (CA)", width=15)
    else:
        # Show only from and to
        table.add_column(f"{from_std.upper()}", width=20)
        table.add_column(f"{to_std.upper()}", width=20)
        table.add_column("Notes", width=40)
    
    for item in items:
        concept_name = item['concept']
        mappings = item['mappings']
        notes = item.get('notes', '')
        
        if show_all or not (from_std and to_std):
            # Display all four columns
            cps230_refs = '\n'.join(mappings.get('cps230', [])) or '[dim]—[/dim]'
            sr117_refs = '\n'.join(mappings.get('sr117', [])) or '[dim]—[/dim]'
            euaiact_refs = '\n'.join(mappings.get('euaiact', [])) or '[dim]—[/dim]'
            osfie23_refs = '\n'.join(mappings.get('osfie23', [])) or '[dim]—[/dim]'
            
            table.add_row(
                concept_name,
                cps230_refs,
                sr117_refs,
                euaiact_refs,
                osfie23_refs
            )
        else:
            # Display from -> to with notes
            from_refs = '\n'.join(mappings.get(from_std, [])) or '[dim]—[/dim]'
            to_refs = '\n'.join(mappings.get(to_std, [])) or '[dim]—[/dim]'
            
            # Truncate notes if too long
            if len(notes) > 100:
                notes = notes[:97] + "..."
            
            table.add_row(concept_name, from_refs, to_refs, notes)
    
    console.print(table)
    console.print(f"\n[dim]Total concepts: {len(items)}[/dim]")


def _display_crosswalk_markdown(items: List[Dict], from_std: Optional[str], to_std: Optional[str], metadata: Dict, show_all: bool):
    """Display crosswalk in markdown format suitable for documentation"""
    
    # Print title and metadata
    print("# Cross-Standard Compliance Crosswalk\n")
    print(f"**Version:** {metadata.get('version', 'unknown')}  ")
    print(f"**Created:** {metadata.get('created', 'unknown')}  ")
    print(f"**Concepts Mapped:** {metadata.get('concepts_mapped', len(items))}  \n")
    
    print("## Standards Covered\n")
    for std in metadata.get('standards_covered', []):
        print(f"- **{std['name']}** ({std['jurisdiction']}): {std['full_name']} — {std['version']}")
    
    print("\n## Mappings\n")
    
    if show_all or not (from_std and to_std):
        # Full table with all four standards
        print("| Concept | CPS 230 (AU) | SR 11-7 (US) | EU AI Act (EU) | OSFI E-23 (CA) |")
        print("|---------|--------------|--------------|----------------|----------------|")
        
        for item in items:
            concept_name = item['concept']
            mappings = item['mappings']
            
            cps230_refs = '<br>'.join(mappings.get('cps230', [])) or '—'
            sr117_refs = '<br>'.join(mappings.get('sr117', [])) or '—'
            euaiact_refs = '<br>'.join(mappings.get('euaiact', [])) or '—'
            osfie23_refs = '<br>'.join(mappings.get('osfie23', [])) or '—'
            
            print(f"| {concept_name} | {cps230_refs} | {sr117_refs} | {euaiact_refs} | {osfie23_refs} |")
    
    else:
        # Two-column from -> to with descriptions
        print(f"### {from_std.upper()} → {to_std.upper()}\n")
        print(f"| Concept | {from_std.upper()} | {to_std.upper()} | Notes |")
        print("|---------|" + "-" * (len(from_std) + 3) + "|" + "-" * (len(to_std) + 3) + "|-------|")
        
        for item in items:
            concept_name = item['concept']
            mappings = item['mappings']
            notes = item.get('notes', '')
            
            from_refs = '<br>'.join(mappings.get(from_std, [])) or '—'
            to_refs = '<br>'.join(mappings.get(to_std, [])) or '—'
            
            print(f"| {concept_name} | {from_refs} | {to_refs} | {notes} |")
    
    # Print footer notes
    notes_text = metadata.get('notes', '')
    if notes_text:
        print("\n## Notes\n")
        print(notes_text)


# ----- Docs subcommand (dbt-style) -----

docs_app = typer.Typer(help="Generate documentation and compliance reports")
app.add_typer(docs_app, name="docs")


@docs_app.command("generate")
def docs_generate(
    model: str = typer.Argument(None, help="Model name"),
    select: str = typer.Option(None, "--select", "-s", help="Model selection criteria"),
    compliance: str = typer.Option(
        None, "--compliance", "-c",
        help="Compliance standard, e.g. standard:cps230"
    ),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use"),
):
    """Generate documentation, optionally with compliance reporting.

    Examples:

        mrm docs generate ccr_monte_carlo --compliance standard:cps230

        mrm docs generate --select ccr_monte_carlo --compliance standard:sr117

        mrm docs generate ccr_monte_carlo -c standard:cps230 -o report.md
    """
    try:
        project = Project.load(profile=profile)

        # Support both positional model argument and --select option
        model_selector = model or select
        if not model_selector:
            console.print(
                "Error: Must specify a model either as an argument or via --select",
                style="red"
            )
            console.print("\nExamples:")
            console.print("  mrm docs generate ccr_monte_carlo --compliance standard:cps230")
            console.print("  mrm docs generate --select ccr_monte_carlo --compliance standard:sr117")
            raise typer.Exit(1)

        model_configs = project.select_models(models=model_selector, select=select if not model else None)
        if not model_configs:
            console.print(f"Model not found: {model}", style="red")
            raise typer.Exit(1)

        model_config = model_configs[0]
        model_name = model_config['model']['name']

        if not compliance:
            console.print(f"Model: [bold]{model_name}[/bold]")
            console.print("No --compliance flag; basic docs only.")
            console.print("Use --compliance standard:<name> for compliance reports.")
            return

        # Parse standard:<name> syntax
        if ":" in compliance:
            prefix, standard_name = compliance.split(":", 1)
            if prefix != "standard":
                console.print(
                    f"Invalid compliance format '{compliance}'. "
                    "Use standard:<name> (e.g. standard:cps230)",
                    style="red",
                )
                raise typer.Exit(1)
        else:
            standard_name = compliance

        console.print(
            f"Generating compliance report ({standard_name}) "
            f"for: [bold]{model_name}[/bold]\n"
        )

        # Run tests
        runner = TestRunner(project.config, project.backend, project.catalog)
        results = runner.run_tests([model_config])
        model_results = results.get(model_name, {})
        test_results = model_results.get('test_results', {})

        # Evaluate triggers
        trigger_events = []
        triggers_cfg = model_config.get('triggers', [])
        if triggers_cfg:
            from mrm.core.triggers import ValidationTriggerEngine
            trigger_engine = ValidationTriggerEngine()
            events = trigger_engine.evaluate(
                model_name=model_name,
                trigger_configs=triggers_cfg,
                test_results=test_results,
            )
            trigger_events = [e.to_dict() for e in events]

        # Generate compliance report via the generic entry point
        from mrm.compliance.report_generator import generate_compliance_report

        output_path = (
            Path(output) if output
            else Path(f"reports/{model_name}_{standard_name}_report.md")
        )

        report_text = generate_compliance_report(
            standard_name=standard_name,
            model_name=model_name,
            model_config=model_config,
            test_results=test_results,
            trigger_events=trigger_events,
            output_path=output_path,
        )

        console.print(f"[green]Report generated: {output_path}[/green]")
        console.print(f"Report size: {len(report_text)} characters")

        _display_test_results(results)

        if trigger_events:
            console.print(f"\n[yellow]{len(trigger_events)} trigger(s) fired[/yellow]")
            for te in trigger_events:
                console.print(f"  - [{te['trigger_type']}] {te['reason']}")

    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except typer.Exit:
        # Re-raise typer.Exit to allow clean exit
        raise
    except Exception as e:
        console.print(f" Error generating report: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@docs_app.command("list-standards")
def docs_list_standards():
    """List available compliance standards"""
    from mrm.compliance.registry import compliance_registry
    compliance_registry.load_builtin_standards()
    standards = compliance_registry.list_standards()

    if not standards:
        console.print("No compliance standards available", style="yellow")
        return

    table = Table(title="Available Compliance Standards", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name")
    table.add_column("Jurisdiction")
    table.add_column("Version")

    for name in standards:
        cls = compliance_registry.get(name)
        table.add_row(name, cls.display_name, cls.jurisdiction, cls.version)

    console.print(table)


@docs_app.command("crosswalk")
def docs_crosswalk(
    from_std: str = typer.Option(None, "--from", help="Source standard (e.g. cps230)"),
    to_std: str = typer.Option(None, "--to", help="Target standard (e.g. sr117)"),
    concept: str = typer.Option(None, "--concept", help="Filter by concept name"),
    show_all: bool = typer.Option(False, "--all", help="Show all mappings (full crosswalk matrix)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table or markdown")
):
    """Display cross-standard compliance crosswalk

    Examples:

        mrm docs crosswalk --from cps230 --to sr117

        mrm docs crosswalk --from euaiact --to osfie23 --concept "Validation"

        mrm docs crosswalk --all

        mrm docs crosswalk --all --format markdown > crosswalk.md
    """
    import yaml
    from pathlib import Path as PathLib
    
    try:
        # Load crosswalk YAML
        crosswalk_path = PathLib(__file__).parent.parent / "compliance" / "crosswalks" / "standards.yaml"
        
        if not crosswalk_path.exists():
            console.print(f"Crosswalk file not found: {crosswalk_path}", style="red")
            raise typer.Exit(1)
        
        with open(crosswalk_path, 'r') as f:
            data = yaml.safe_load(f)
        
        crosswalk_items = data.get('crosswalk', [])
        metadata = data.get('metadata', {})
        
        if not crosswalk_items:
            console.print("No crosswalk data found", style="yellow")
            raise typer.Exit(1)
        
        # Standard name mapping (handle both short and display names)
        standard_map = {
            'cps230': 'cps230',
            'sr117': 'sr117',
            'sr11-7': 'sr117',
            'euaiact': 'euaiact',
            'eu_ai_act': 'euaiact',
            'osfie23': 'osfie23',
            'osfi_e23': 'osfie23',
        }
        
        # Normalize standard names
        from_std_normalized = standard_map.get(from_std.lower(), from_std.lower()) if from_std else None
        to_std_normalized = standard_map.get(to_std.lower(), to_std.lower()) if to_std else None
        
        # Filter items
        filtered_items = crosswalk_items
        
        if concept:
            concept_lower = concept.lower()
            filtered_items = [
                item for item in filtered_items
                if concept_lower in item['concept'].lower() or 
                   concept_lower in item.get('description', '').lower()
            ]
        
        if from_std and not show_all:
            # Filter to items that have mappings in source standard
            filtered_items = [
                item for item in filtered_items
                if item['mappings'].get(from_std_normalized, [])
            ]
        
        if to_std and not show_all:
            # Filter to items that have mappings in target standard
            filtered_items = [
                item for item in filtered_items
                if item['mappings'].get(to_std_normalized, [])
            ]
        
        if not filtered_items:
            console.print("No matching mappings found", style="yellow")
            raise typer.Exit(1)
        
        # Display results
        if format == "markdown":
            _display_crosswalk_markdown(
                filtered_items,
                from_std_normalized,
                to_std_normalized,
                metadata,
                show_all
            )
        else:
            _display_crosswalk_table(
                filtered_items,
                from_std_normalized,
                to_std_normalized,
                show_all
            )
        
        # Display metadata footer
        if not show_all:
            console.print(f"\n[dim]Crosswalk version: {metadata.get('version', 'unknown')}[/dim]")
            console.print(f"[dim]Concepts mapped: {metadata.get('concepts_mapped', len(crosswalk_items))}[/dim]")
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error loading crosswalk: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command(deprecated=True)
def report(
    model: str = typer.Argument(..., help="Model name"),
    format: str = typer.Option("markdown", "--format", "-f", help="Report format"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use"),
):
    """[DEPRECATED] Use 'mrm docs generate --compliance standard:cps230' instead"""
    import warnings
    warnings.warn(
        "The 'report' command is deprecated. "
        "Use 'mrm docs generate <model> --compliance standard:cps230' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    console.print(
        "[yellow]DEPRECATED: Use 'mrm docs generate <model> "
        "--compliance standard:cps230' instead[/yellow]\n"
    )
    docs_generate(
        model=model, compliance="standard:cps230",
        format=format, output=output, profile=profile,
    )


# ----- Triggers subcommand -----
triggers_app = typer.Typer(help="Manage validation triggers")
app.add_typer(triggers_app, name="triggers")


@triggers_app.command("check")
def triggers_check(
    model: str = typer.Argument(..., help="Model name to check triggers for"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use"),
):
    """Evaluate validation triggers for a model"""
    try:
        project = Project.load(profile=profile)
        model_configs = project.select_models(models=model)

        if not model_configs:
            console.print(f"Model not found: {model}", style="red")
            raise typer.Exit(1)

        model_config = model_configs[0]
        model_name = model_config['model']['name']
        triggers_cfg = model_config.get('triggers', [])

        if not triggers_cfg:
            console.print(f"No triggers configured for {model_name}", style="yellow")
            raise typer.Exit(0)

        from mrm.core.triggers import ValidationTriggerEngine
        engine = ValidationTriggerEngine()
        events = engine.evaluate(model_name=model_name, trigger_configs=triggers_cfg)

        if events:
            table = Table(title=f"Fired Triggers - {model_name}", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Type")
            table.add_column("Reason")
            table.add_column("Compliance Ref")
            table.add_column("Status")

            for e in events:
                table.add_row(
                    e.trigger_id,
                    e.trigger_type.value,
                    e.reason,
                    e.compliance_reference,
                    e.status.value,
                )
            console.print(table)
        else:
            console.print(f"[green]No triggers fired for {model_name}[/green]")

    except Exception as e:
        console.print(f" Error checking triggers: {e}", style="red")
        raise typer.Exit(1)


@triggers_app.command("list")
def triggers_list(
    model: str = typer.Option(None, "--model", "-m", help="Filter by model name"),
):
    """List all trigger events"""
    try:
        from mrm.core.triggers import ValidationTriggerEngine
        engine = ValidationTriggerEngine()
        events = engine.get_all_events(model_name=model)

        if not events:
            console.print("No trigger events found", style="yellow")
            raise typer.Exit(0)

        table = Table(title="Trigger Events", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Model")
        table.add_column("Type")
        table.add_column("Fired At")
        table.add_column("Reason")
        table.add_column("Status")

        for e in events:
            status_style = {
                "fired": "red",
                "acknowledged": "yellow",
                "resolved": "green",
            }.get(e.status.value, "white")

            table.add_row(
                e.trigger_id,
                e.model_name,
                e.trigger_type.value,
                e.fired_at[:19],
                e.reason[:50],
                f"[{status_style}]{e.status.value}[/{status_style}]",
            )
        console.print(table)

    except Exception as e:
        console.print(f" Error listing triggers: {e}", style="red")
        raise typer.Exit(1)


@triggers_app.command("resolve")
def triggers_resolve(
    model: str = typer.Argument(..., help="Model name to resolve triggers for"),
):
    """Resolve all active triggers for a model (after re-validation)"""
    try:
        from mrm.core.triggers import ValidationTriggerEngine
        engine = ValidationTriggerEngine()
        active = engine.get_active_triggers(model_name=model)

        if not active:
            console.print(f"No active triggers for {model}", style="green")
            raise typer.Exit(0)

        engine.resolve_model(model)
        console.print(
            f"[green]Resolved {len(active)} trigger(s) for {model}[/green]"
        )

    except Exception as e:
        console.print(f" Error resolving triggers: {e}", style="red")
        raise typer.Exit(1)


# ----- Evidence subcommand -----
evidence_app = typer.Typer(help="Manage immutable evidence vault")
app.add_typer(evidence_app, name="evidence")


@evidence_app.command("freeze")
def evidence_freeze(
    model: str = typer.Argument(..., help="Model name to create evidence for"),
    backend: str = typer.Option("local", "--backend", "-b", help="Backend: local or s3"),
    bucket: str = typer.Option(None, "--bucket", help="S3 bucket name (for S3 backend)"),
    retention: int = typer.Option(2555, "--retention", "-r", help="Retention period in days"),
    created_by: str = typer.Option(None, "--created-by", help="User identifier (email)"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use"),
):
    """Freeze validation results as immutable evidence packet
    
    Examples:
    
        mrm evidence freeze ccr_monte_carlo --backend local
        
        mrm evidence freeze ccr_monte_carlo --backend s3 --bucket my-evidence --retention 2555
    """
    try:
        from pathlib import Path as PathLib
        import getpass
        import os
        from mrm.evidence.packet import EvidencePacket
        from mrm.evidence.backends.local import LocalFilesystemBackend
        
        # Load project
        project = Project.load(profile=profile)
        model_configs = project.select_models(models=model)
        
        if not model_configs:
            console.print(f"Model not found: {model}", style="red")
            raise typer.Exit(1)
        
        model_config = model_configs[0]
        model_name = model_config['model']['name']
        model_info = model_config['model']
        
        console.print(f"Creating evidence packet for: [bold]{model_name}[/bold]")
        
        # Get model artifact path
        location = model_info.get('location', {})
        if isinstance(location, str):
            if location.startswith('file/'):
                model_path = location[5:]
            else:
                model_path = location
        else:
            model_path = location.get('path')
        
        if not model_path:
            console.print("Model location/path not found in config", style="red")
            raise typer.Exit(1)
        
        # Make path absolute
        if not PathLib(model_path).is_absolute():
            model_path = str(project.root_path / model_path)
        
        model_artifact = PathLib(model_path)
        if not model_artifact.exists():
            console.print(f"Model artifact not found: {model_path}", style="red")
            raise typer.Exit(1)
        
        # Run tests to get current results
        console.print("Running validation tests...")
        from mrm.engine.runner import TestRunner
        runner = TestRunner(project.config, project.backend, project.catalog)
        results = runner.run_tests([model_config])
        
        model_results = results.get(model_name, {})
        test_results_raw = model_results.get('test_results', {})
        
        if not test_results_raw:
            console.print("No test results available", style="yellow")
            raise typer.Exit(1)
        
        # Convert TestResult objects to dicts
        test_results = {}
        for test_name, test_result in test_results_raw.items():
            if hasattr(test_result, 'to_dict'):
                test_results[test_name] = test_result.to_dict()
            else:
                test_results[test_name] = test_result
        
        # Get compliance mappings (these come from the model config)
        # For now, we'll extract from model's configured tests
        compliance_mappings = {}
        tests_cfg = model_config.get('tests', [])
        for test_cfg in tests_cfg:
            if isinstance(test_cfg, dict):
                test_compliance = test_cfg.get('compliance', {})
                for standard, paragraphs in test_compliance.items():
                    if standard not in compliance_mappings:
                        compliance_mappings[standard] = []
                    if isinstance(paragraphs, list):
                        compliance_mappings[standard].extend(paragraphs)
                    else:
                        compliance_mappings[standard].append(paragraphs)
        
        # Get created_by (user email/username)
        if not created_by:
            created_by = os.environ.get('USER', getpass.getuser())
        
        # Initialize backend
        if backend == 'local':
            evidence_dir = project.root_path / "evidence"
            backend_impl = LocalFilesystemBackend(evidence_dir)
            
        elif backend == 's3':
            if not bucket:
                console.print("--bucket required for S3 backend", style="red")
                raise typer.Exit(1)
            
            try:
                from mrm.evidence.backends.s3_object_lock import S3ObjectLockBackend
            except ImportError:
                console.print(
                    "S3 backend requires boto3: pip install boto3",
                    style="red"
                )
                raise typer.Exit(1)
            
            backend_impl = S3ObjectLockBackend(bucket=bucket)
        
        else:
            console.print(f"Unknown backend: {backend}", style="red")
            raise typer.Exit(1)
        
        # Get prior packet (for hash chain)
        prior_packet = backend_impl.get_latest_packet(model_name)
        
        # Create evidence packet
        console.print("Creating evidence packet...")
        packet = EvidencePacket.create(
            model_name=model_name,
            model_version=model_info.get('version', '1.0'),
            model_artifact_path=model_artifact,
            test_results=test_results,
            compliance_mappings=compliance_mappings,
            created_by=created_by,
            prior_packet=prior_packet,
            metadata={
                'profile': profile,
                'risk_tier': model_info.get('risk_tier'),
                'owner': model_info.get('owner')
            }
        )
        
        # Verify packet before freezing
        if not packet.verify_hash():
            console.print("Packet hash verification failed", style="red")
            raise typer.Exit(1)
        
        # Freeze packet
        console.print(f"Freezing packet with {backend} backend...")
        uri = backend_impl.freeze(packet, retention_days=retention)
        
        console.print(f"\n[green]✓ Evidence packet frozen successfully[/green]")
        console.print(f"  Packet ID: {packet.packet_id}")
        console.print(f"  URI: {uri}")
        console.print(f"  Content Hash: {packet.content_hash}")
        console.print(f"  Model Artifact Hash: {packet.model_artifact_hash}")
        
        if prior_packet:
            console.print(f"  Prior Packet: {prior_packet.packet_id}")
            console.print(f"  Chain Length: {len(backend_impl.list_packets(model_name=model_name))}")
        else:
            console.print(f"  [yellow]First packet in chain[/yellow]")
        
        console.print(f"\nNext steps:")
        console.print(f"  mrm evidence verify {uri}")
        console.print(f"  mrm evidence list --model {model_name}")
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error freezing evidence: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@evidence_app.command("verify")
def evidence_verify(
    uri: str = typer.Argument(..., help="Evidence packet URI"),
    chain: bool = typer.Option(True, "--chain/--no-chain", help="Verify full hash chain"),
):
    """Verify evidence packet integrity and hash chain
    
    Examples:
    
        mrm evidence verify file:///path/to/packets.jsonl#packet-id
        
        mrm evidence verify s3://bucket/evidence/model/packet-id.json --chain
    """
    try:
        from pathlib import Path as PathLib
        from mrm.evidence.backends.local import LocalFilesystemBackend
        
        # Determine backend from URI
        if uri.startswith('file://'):
            # Parse path from URI
            path_part = uri[7:].split('#')[0]
            evidence_dir = PathLib(path_part).parent.parent
            backend = LocalFilesystemBackend(evidence_dir, warn_on_use=False)
            
        elif uri.startswith('s3://'):
            try:
                from mrm.evidence.backends.s3_object_lock import S3ObjectLockBackend
            except ImportError:
                console.print(
                    "S3 backend requires boto3: pip install boto3",
                    style="red"
                )
                raise typer.Exit(1)
            
            # Parse bucket from URI
            bucket = uri.split('/')[2]
            backend = S3ObjectLockBackend(bucket=bucket)
        
        else:
            console.print(f"Unknown URI scheme: {uri}", style="red")
            raise typer.Exit(1)
        
        # Verify packet
        console.print(f"Verifying: {uri}")
        result = backend.verify(uri, verify_chain=chain)
        
        if result['valid']:
            console.print(f"\n[green]✓ Verification passed[/green]")
            console.print(f"  Reason: {result['reason']}")
            
            if 'packet_count' in result:
                console.print(f"  Chain length: {result['packet_count']} packets")
                console.print(f"  First packet: {result.get('first_packet', 'N/A')}")
                console.print(f"  Latest packet: {result.get('latest_packet', 'N/A')}")
            
            if 'retention_info' in result:
                retention = result['retention_info']
                if 'error' not in retention:
                    console.print(f"  Retention mode: {retention.get('retention_mode', 'N/A')}")
                    console.print(f"  Retain until: {retention.get('retain_until', 'N/A')}")
        
        else:
            console.print(f"\n[red]✗ Verification failed[/red]")
            console.print(f"  Reason: {result['reason']}")
            
            if 'packet_id' in result:
                console.print(f"  Packet ID: {result['packet_id']}")
            
            raise typer.Exit(1)
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error verifying evidence: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@evidence_app.command("list")
def evidence_list(
    model: str = typer.Option(None, "--model", "-m", help="Filter by model name"),
    backend: str = typer.Option("local", "--backend", "-b", help="Backend: local or s3"),
    bucket: str = typer.Option(None, "--bucket", help="S3 bucket name (for S3 backend)"),
    profile: str = typer.Option("dev", "--profile", "-p", help="Profile to use"),
):
    """List evidence packets
    
    Examples:
    
        mrm evidence list --model ccr_monte_carlo
        
        mrm evidence list --backend s3 --bucket my-evidence
    """
    try:
        from pathlib import Path as PathLib
        from mrm.evidence.backends.local import LocalFilesystemBackend
        
        # Initialize backend
        if backend == 'local':
            project = Project.load(profile=profile)
            evidence_dir = project.root_path / "evidence"
            backend_impl = LocalFilesystemBackend(evidence_dir, warn_on_use=False)
            
        elif backend == 's3':
            if not bucket:
                console.print("--bucket required for S3 backend", style="red")
                raise typer.Exit(1)
            
            try:
                from mrm.evidence.backends.s3_object_lock import S3ObjectLockBackend
            except ImportError:
                console.print(
                    "S3 backend requires boto3: pip install boto3",
                    style="red"
                )
                raise typer.Exit(1)
            
            backend_impl = S3ObjectLockBackend(bucket=bucket)
        
        else:
            console.print(f"Unknown backend: {backend}", style="red")
            raise typer.Exit(1)
        
        # List packets
        packets = backend_impl.list_packets(model_name=model)
        
        if not packets:
            console.print("No evidence packets found", style="yellow")
            raise typer.Exit(0)
        
        # Display table
        table = Table(title=f"Evidence Packets ({backend})", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Version")
        table.add_column("Packet ID")
        table.add_column("Timestamp")
        table.add_column("Created By")
        
        for packet in packets:
            table.add_row(
                packet.get('model_name', '')[:20],
                packet.get('model_version', '')[:10],
                packet.get('packet_id', '')[:12] + '...',
                packet.get('timestamp', '')[:19],
                packet.get('created_by', '')[:20]
            )
        
        console.print(table)
        console.print(f"\n[dim]Total: {len(packets)} packet(s)[/dim]")
    
    except FileNotFoundError as e:
        console.print(f" {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f" Error listing evidence: {e}", style="red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
