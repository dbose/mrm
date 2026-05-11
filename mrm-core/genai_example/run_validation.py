"""
Run validation for GenAI RAG Customer Service example.

This script demonstrates end-to-end validation of an LLM-based system
using mrm-core's genai test suite.
"""

from pathlib import Path
import sys

# Add mrm to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from mrm.core.project import Project
from mrm.engine.runner import TestRunner
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Run validation for RAG assistant model."""
    console.print("\n[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print("[bold blue]GenAI RAG Customer Service - Validation[/bold blue]")
    console.print("[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print()
    
    # Load project
    project_path = Path(__file__).parent
    console.print(f"[dim]Loading project from: {project_path}[/dim]")
    project = Project.load(project_path)
    
    # Get model
    model_name = "rag_assistant"
    models = project.select_models(models=model_name)
    
    if not models:
        console.print(f"[red]✗ Model '{model_name}' not found[/red]")
        return
    
    model_config = models[0]
    console.print(f"[dim]Model: {model_config.get('name', model_name)} v{model_config.get('version', '1.0.0')}[/dim]\n")
    
    # Check API key
    import os
    location = model_config.get('location', {})
    provider = location.get('provider', 'openai')
    env_var = f"{provider.upper()}_API_KEY"
    
    if not os.getenv(env_var):
        console.print(f"[yellow]⚠️  Warning: {env_var} not set[/yellow]")
        console.print(f"[yellow]   Set with: export {env_var}='your-key'[/yellow]")
        console.print(f"[yellow]   Continuing anyway (some tests may fail)...[/yellow]\n")
    
    # Initialize test runner
    runner = TestRunner(project.config, project.backend, project.catalog)
    
    # Run tests
    console.print("[bold]Running GenAI Test Suite[/bold]")
    console.print("[dim]This may take several minutes depending on API latency...[/dim]\n")
    
    results = runner.run_tests(
        model_configs=[model_config],
        test_selection=None,  # Run all configured tests
        fail_fast=False
    )
    
    # Print results summary
    console.print("\n" + "=" * 60)
    console.print("[bold blue]Test Results Summary[/bold blue]")
    console.print("=" * 60 + "\n")
    
    model_results = results.get(model_name, [])
    
    # Check if error at model level
    if isinstance(model_results, dict) and 'error' in model_results:
        console.print(f"[red]Error: {model_results['error']}[/red]")
        return
    
    # Get test results
    if isinstance(model_results, dict):
        test_results = model_results.get('test_results', {})
        all_passed = model_results.get('all_passed', False)
    else:
        console.print("[yellow]Unexpected results format[/yellow]")
        return
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", width=35)
    table.add_column("Status", width=10)
    table.add_column("Details", width=40)
    
    passed = 0
    failed = 0
    errors = 0
    
    for test_name, result in test_results.items():
        if isinstance(result, str):
            # Error message
            table.add_row(test_name, "[red]✗ ERROR[/red]", result[:70])
            errors += 1
        else:
            status_color = "green" if result.passed else "red"
            status_text = "✓ PASS" if result.passed else "✗ FAIL"
            
            details = result.failure_reason or ""
            if hasattr(result, 'details') and result.details:
                # Add key details
                detail_str = ", ".join([f"{k}: {v}" for k, v in list(result.details.items())[:2]])
                details = f"{details[:30]}... | {detail_str}" if len(details) > 30 else detail_str
            
            table.add_row(
                test_name,
                f"[{status_color}]{status_text}[/{status_color}]",
                details[:70]
            )
            
            if result.passed:
                passed += 1
            else:
                failed += 1
    
    console.print(table)
    console.print()
    
    # Summary stats
    total = passed + failed + errors
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    console.print(f"[bold]Total Tests:[/bold] {total}")
    console.print(f"[green]Passed:[/green] {passed}")
    console.print(f"[red]Failed:[/red] {failed}")
    if errors > 0:
        console.print(f"[yellow]Errors:[/yellow] {errors}")
    console.print(f"[bold]Pass Rate:[/bold] {pass_rate:.1f}%\n")
    
    # Check triggers
    console.print("[bold blue]Checking Validation Triggers[/bold blue]")
    # TODO: Implement trigger checking  
    # from mrm.core.triggers import check_triggers
    # trigger_results = check_triggers(project, model_config, model_results)
    
    console.print("[yellow]Note: Trigger checking not yet implemented[/yellow]")
    trigger_results = []
    
    if trigger_results:
        console.print("[yellow]⚠️  Triggers activated:[/yellow]")
        for trigger in trigger_results:
            console.print(f"   • {trigger['type']}: {trigger['description']}")
    else:
        console.print("[green]✓ No triggers activated[/green]")
    
    console.print()
    
    # Generate compliance reports
    console.print("[bold blue]Generating Compliance Reports[/bold blue]")
    
    for standard in ['cps230', 'euaiact', 'sr117']:  # Fixed: euaiact not eu_ai_act
        try:
            from mrm.compliance.report_generator import generate_compliance_report
            
            report = generate_compliance_report(
                standard_name=standard,
                model_name=model_name,
                model_config=model_config,
                test_results=test_results
            )
            
            report_path = project_path / "reports" / f"{model_name}_{standard}_report.md"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            console.print(f"[green]✓[/green] Generated {standard} report: {report_path.name}")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not generate {standard} report: {e}[/yellow]")
    
    console.print()
    
    # Save evidence
    console.print("[bold blue]Freezing Evidence[/bold blue]")
    try:
        from mrm.evidence.packet import EvidencePacket
        from mrm.evidence.backends.local import LocalFilesystemBackend
        import uuid
        from datetime import datetime
        
        backend = LocalFilesystemBackend(evidence_dir=project_path / "evidence")
        
        # Extract model metadata from config dict
        model_meta = model_config.get('model', {})
        model_name_str = model_meta.get('name', model_name)
        
        # Get prior packet hash for chain
        try:
            prior_packet = backend.get_latest_packet(model_name_str)
            prior_hash = prior_packet.content_hash if prior_packet else None
        except:
            prior_hash = None
        
        # Create evidence packet
        packet = EvidencePacket(
            packet_id=str(uuid.uuid4()),
            model_name=model_name_str,
            model_version=model_meta.get('version', '1.0.0'),
            model_artifact_hash='llm_endpoint',  # LLM endpoint has no local artifact
            test_results={k: v.to_dict() if hasattr(v, 'to_dict') else str(v) 
                         for k, v in test_results.items()},
            compliance_mappings=model_config.get('compliance', {}),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            created_by="genai_validation_script",
            prior_packet_hash=prior_hash
        )
        
        uri = backend.freeze(packet)
        console.print(f"[green]✓[/green] Evidence frozen: {uri}")
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not freeze evidence: {e}[/yellow]")
    
    console.print("\n" + "=" * 60)
    console.print("[bold blue]✓ Validation Complete[/bold blue]")
    console.print("=" * 60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
