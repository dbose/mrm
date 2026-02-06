"""
Example: Model Dependencies and DAG

This demonstrates:
1. Building a hierarchy of models with dependencies
2. Using ref() to reference models
3. DAG-based test execution
4. Graph operators for selection
5. HuggingFace integration
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import sys

# Add mrm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_dag_example():
    """Create example project with model dependencies"""
    
    print("=" * 70)
    print("MRM Core - Model DAG and ref() Example")
    print("=" * 70)
    
    # Create base dataset
    print("\n1. Creating synthetic credit dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(20)]
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['target'] = y_val
    
    # Initialize project
    print("\n2. Initializing project with model hierarchy...")
    from mrm.core.init import initialize_project
    
    project_name = "credit_dag_example"
    project_path = Path(project_name)
    
    if project_path.exists():
        import shutil
        shutil.rmtree(project_path)
    
    initialize_project(project_name, backend="local")
    
    # Create models directory structure
    (project_path / "models" / "base").mkdir(parents=True, exist_ok=True)
    (project_path / "models" / "composite").mkdir(parents=True, exist_ok=True)
    (project_path / "models" / "ensemble").mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    data_dir = project_path / "data"
    train_df.to_csv(data_dir / "training.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)
    
    # Create Level 0: Base models (no dependencies)
    print("\n3. Creating base models (Level 0)...")
    
    # PD Model
    pd_model = LogisticRegression(random_state=42, max_iter=1000)
    pd_model.fit(X_train, y_train)
    
    with open(project_path / "models" / "pd_model.pkl", 'wb') as f:
        pickle.dump(pd_model, f)
    
    pd_config = """model:
  name: pd_model
  version: 1.0.0
  risk_tier: tier_1
  description: "Probability of Default - Base Model"
  
  location:
    type: file
    path: models/pd_model.pkl

datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.70
  - test: model.Gini
    config:
      min_score: 0.40
"""
    
    with open(project_path / "models" / "base" / "pd_model.yml", 'w') as f:
        f.write(pd_config)
    
    # LGD Model
    lgd_model = RandomForestClassifier(n_estimators=50, random_state=42)
    lgd_model.fit(X_train, y_train)
    
    with open(project_path / "models" / "lgd_model.pkl", 'wb') as f:
        pickle.dump(lgd_model, f)
    
    lgd_config = """model:
  name: lgd_model
  version: 1.0.0
  risk_tier: tier_1
  description: "Loss Given Default - Base Model"
  
  location:
    type: file
    path: models/lgd_model.pkl

datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.70
  - test: model.Accuracy
    config:
      min_score: 0.70
"""
    
    with open(project_path / "models" / "base" / "lgd_model.yml", 'w') as f:
        f.write(lgd_config)
    
    # Create Level 1: Composite model (depends on base models)
    print("\n4. Creating composite model (Level 1)...")
    
    # Simple ensemble that averages predictions
    class ExpectedLossModel:
        """Combines PD and LGD models"""
        def __init__(self, pd_model, lgd_model):
            self.pd_model = pd_model
            self.lgd_model = lgd_model
        
        def predict_proba(self, X):
            pd_probs = self.pd_model.predict_proba(X)
            lgd_probs = self.lgd_model.predict_proba(X)
            # Average the probabilities
            avg_probs = (pd_probs + lgd_probs) / 2
            return avg_probs
        
        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] > 0.5).astype(int)
    
    el_model = ExpectedLossModel(pd_model, lgd_model)
    
    with open(project_path / "models" / "expected_loss.pkl", 'wb') as f:
        pickle.dump(el_model, f)
    
    el_config = """model:
  name: expected_loss
  version: 1.0.0
  risk_tier: tier_1
  description: "Expected Loss - Composite Model"
  
  # Dependencies!
  depends_on:
    - pd_model
    - lgd_model
  
  location:
    type: file
    path: models/expected_loss.pkl

datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.75
  - test: model.Precision
    config:
      min_score: 0.70
"""
    
    with open(project_path / "models" / "composite" / "expected_loss.yml", 'w') as f:
        f.write(el_config)
    
    # Create Level 2: Ensemble (depends on all previous)
    print("\n5. Creating ensemble model (Level 2)...")
    
    class EnsembleModel:
        """Final ensemble combining all base models"""
        def __init__(self, models):
            self.models = models
        
        def predict_proba(self, X):
            all_probs = [m.predict_proba(X) for m in self.models if hasattr(m, 'predict_proba')]
            if not all_probs:
                all_probs = [[0.5, 0.5]] * len(X)
            avg_probs = np.mean(all_probs, axis=0)
            return avg_probs
        
        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] > 0.5).astype(int)
    
    ensemble = EnsembleModel([pd_model, lgd_model, el_model])
    
    with open(project_path / "models" / "ensemble_model.pkl", 'wb') as f:
        pickle.dump(ensemble, f)
    
    ensemble_config = """model:
  name: ensemble_model
  version: 1.0.0
  risk_tier: tier_1
  description: "Final Ensemble Model"
  
  # Depends on everything
  depends_on:
    - pd_model
    - lgd_model
    - expected_loss
  
  location:
    type: file
    path: models/ensemble_model.pkl

datasets:
  validation:
    type: csv
    path: data/validation.csv

tests:
  - test: model.ROCAUC
    config:
      min_score: 0.80
  - test: model.Gini
    config:
      min_score: 0.60
"""
    
    with open(project_path / "models" / "ensemble" / "ensemble.yml", 'w') as f:
        f.write(ensemble_config)
    
    # Demonstrate DAG functionality
    print("\n6. Building and visualizing DAG...")
    
    import os
    os.chdir(project_path)
    
    from mrm.core.project import Project
    from mrm.core.dag import ModelDAG
    
    project = Project.load()
    dag = project.dag
    
    print("\n" + "=" * 70)
    print("MODEL DEPENDENCY GRAPH")
    print("=" * 70)
    print(dag.visualize())
    
    # Show execution levels
    levels = dag.get_execution_levels()
    print("\n" + "=" * 70)
    print("EXECUTION LEVELS (for parallel execution)")
    print("=" * 70)
    for i, level in enumerate(levels):
        print(f"Level {i}: {level}")
    
    # Demonstrate graph operators
    print("\n" + "=" * 70)
    print("GRAPH OPERATOR EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("pd_model", "Just pd_model"),
        ("+pd_model", "pd_model and all upstream"),
        ("pd_model+", "pd_model and all downstream"),
        ("+pd_model+", "pd_model, upstream, and downstream"),
        ("@ensemble_model", "Just ensemble_model"),
        ("+ensemble_model", "ensemble_model and all upstream"),
    ]
    
    for selector, description in examples:
        selected = dag.select_nodes(selector)
        print(f"\n'{selector}' ({description}):")
        print(f"  → {sorted(selected)}")
    
    # Test in dependency order
    print("\n" + "=" * 70)
    print("TESTING IN DEPENDENCY ORDER")
    print("=" * 70)
    
    from mrm.engine.runner import TestRunner
    
    # Test all models in dependency order
    models = project.list_models()
    runner = TestRunner(project.config, project.backend, project.catalog)
    
    print("\nRunning tests for all models in dependency order...")
    results = runner.run_tests(models, threads=1)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"\n{model_name}: ERROR")
            print(f"  {result['error']}")
        else:
            status = " PASSED" if result['all_passed'] else " FAILED"
            print(f"\n{model_name}: {status}")
            print(f"  Tests run: {result['tests_run']}")
            print(f"  Passed: {result['tests_passed']}")
            print(f"  Failed: {result['tests_failed']}")
    
    os.chdir('..')
    
    print("\n" + "=" * 70)
    print("SUCCESS! DAG functionality demonstrated")
    print("=" * 70)
    
    print(f"\nProject created at: {project_path.absolute()}")
    print("\nTry these commands:")
    print(f"  cd {project_name}")
    print("  mrm test                          # Test all in dependency order")
    print("  mrm test --select +expected_loss  # Test expected_loss and dependencies")
    print("  mrm test --select pd_model+       # Test pd_model and downstream")
    print("  mrm debug --show-dag              # View dependency graph")


if __name__ == "__main__":
    create_dag_example()
