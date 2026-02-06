"""
Example usage of MRM Core

This script demonstrates how to:
1. Create a project
2. Define a simple model
3. Run validation tests
4. View results
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os
import sys

# Add mrm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_example_project():
    """Create an example MRM project with a trained model"""
    
    print("=" * 60)
    print("MRM Core - Example Usage")
    print("=" * 60)
    
    # Create synthetic credit risk dataset
    print("\n1. Creating synthetic credit risk dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced
        random_state=42
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create DataFrames
    feature_names = [f'feature_{i}' for i in range(20)]
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['target'] = y_val
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Default rate: {y_val.mean():.2%}")
    
    # Train a simple logistic regression model
    print("\n2. Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    gini = 2 * roc_auc - 1
    
    print(f"   ROC AUC: {roc_auc:.3f}")
    print(f"   Gini: {gini:.3f}")
    
    # Create project directory
    project_name = "credit_risk_example"
    project_path = Path(project_name)
    
    if project_path.exists():
        import shutil
        shutil.rmtree(project_path)
    
    print(f"\n3. Initializing MRM project: {project_name}")
    
    # Use the CLI to initialize
    from mrm.core.init import initialize_project
    initialize_project(project_name, template="credit_risk", backend="local")
    
    # Save model
    print("\n4. Saving model and datasets...")
    models_dir = project_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "credit_scorecard.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save datasets
    data_dir = project_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(data_dir / "training.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)
    
    print("   Model saved to models/credit_scorecard.pkl")
    print("   Data saved to data/")
    
    # Update model config
    print("\n5. Updating model configuration...")
    model_config = f"""# Credit Risk Scorecard Model
model:
  name: credit_scorecard
  version: 1.0.0
  risk_tier: tier_1
  
  description: "Probability of Default model for consumer credit"
  owner: credit_risk_team
  use_case: consumer_lending
  methodology: logistic_regression
  
  location:
    type: file
    path: models/credit_scorecard.pkl
  
datasets:
  training:
    type: csv
    path: data/training.csv
    
  validation:
    type: csv  
    path: data/validation.csv

tests:
  # Data quality tests
  - test: tabular_dataset.MissingValues
    config:
      dataset: validation
      threshold: 0.05
      
  - test: tabular_dataset.ClassImbalance
    config:
      dataset: validation
      target_column: target
      min_ratio: 0.1
      
  - test: tabular_dataset.OutlierDetection
    config:
      dataset: validation
      threshold: 0.15
  
  # Model performance tests
  - test: model.Accuracy
    config:
      dataset: validation
      min_score: 0.70
      
  - test: model.ROCAUC
    config:
      dataset: validation
      min_score: 0.70
      
  - test: model.Gini
    config:
      dataset: validation
      min_score: 0.40
      
  - test: model.Precision
    config:
      dataset: validation
      min_score: 0.65
      
  - test: model.Recall
    config:
      dataset: validation
      min_score: 0.65
"""
    
    model_file = project_path / "models" / "credit_risk" / "scorecard.yml"
    with open(model_file, 'w') as f:
        f.write(model_config)
    
    print("\n6. Running validation tests...")
    
    # Change to project directory
    original_dir = os.getcwd()
    os.chdir(project_path)
    
    try:
        from mrm.core.project import Project
        from mrm.engine.runner import TestRunner
        
        # Load project
        project = Project.load()
        
        # Get model
        models = project.list_models()
        print(f"\n   Found {len(models)} model(s)")
        
        # Run tests
        runner = TestRunner(project.config, project.backend)
        results = runner.run_tests(models, threads=1)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        for model_name, result in results.items():
            print(f"\nModel: {model_name}")
            print(f"Status: {'PASSED ' if result['all_passed'] else 'FAILED '}")
            print(f"Tests run: {result['tests_run']}")
            print(f"Passed: {result['tests_passed']}")
            print(f"Failed: {result['tests_failed']}")
            
            print("\nDetailed results:")
            for test_name, test_result in result['test_results'].items():
                status = "" if test_result.passed else ""
                score_str = f" (score: {test_result.score:.3f})" if test_result.score is not None else ""
                print(f"  {status} {test_name}{score_str}")
                if not test_result.passed and test_result.failure_reason:
                    print(f"    Reason: {test_result.failure_reason}")
        
        print("\n" + "=" * 60)
        print("SUCCESS! MRM Core is working correctly.")
        print("=" * 60)
        
        print(f"\nProject created at: {project_path.absolute()}")
        print("\nNext steps:")
        print(f"  cd {project_name}")
        print("  mrm list models")
        print("  mrm test --models credit_scorecard")
        print("  mrm debug --show-config")
    
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    create_example_project()
