"""Built-in tests for models"""

import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from mrm.tests.base import ModelTest, TestResult
from mrm.tests.library import register_test


@register_test
class ModelAccuracy(ModelTest):
    """Test model accuracy"""
    
    name = "model.Accuracy"
    description = "Measure model classification accuracy"
    tags = ["performance", "classification"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.70, **config) -> TestResult:
        """
        Test model accuracy
        
        Args:
            model: Trained model with predict method
            dataset: pandas DataFrame with features and target
            min_score: Minimum required accuracy
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        # Assume last column is target
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            y_pred = model.predict(X)
            accuracy = accuracy_score(y_true, y_pred)
            
            passed = accuracy >= min_score
            
            return TestResult(
                passed=passed,
                score=float(accuracy),
                details={
                    'accuracy': float(accuracy),
                    'min_score': min_score,
                    'num_samples': len(y_true)
                },
                failure_reason=f"Accuracy {accuracy:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing accuracy: {str(e)}"
            )


@register_test
class ModelROCAUC(ModelTest):
    """Test model ROC AUC score"""
    
    name = "model.ROCAUC"
    description = "Measure model ROC AUC score"
    tags = ["performance", "classification", "discrimination"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.70, **config) -> TestResult:
        """
        Test model ROC AUC
        
        Args:
            model: Trained model with predict_proba method
            dataset: pandas DataFrame with features and target
            min_score: Minimum required ROC AUC
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                # Handle binary and multiclass
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    # Multiclass - use ovr
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    if y_true_bin.shape[1] == 1:
                        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X)
            else:
                return TestResult(
                    passed=False,
                    failure_reason="Model must have predict_proba or decision_function method"
                )
            
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr' if y_pred_proba.ndim > 1 else 'raise')
            
            passed = roc_auc >= min_score
            
            return TestResult(
                passed=passed,
                score=float(roc_auc),
                details={
                    'roc_auc': float(roc_auc),
                    'min_score': min_score,
                    'num_samples': len(y_true)
                },
                failure_reason=f"ROC AUC {roc_auc:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing ROC AUC: {str(e)}"
            )


@register_test
class ModelGini(ModelTest):
    """Test model Gini coefficient"""
    
    name = "model.Gini"
    description = "Measure model discrimination using Gini coefficient"
    tags = ["performance", "classification", "discrimination"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.40, **config) -> TestResult:
        """
        Test model Gini coefficient (2*AUC - 1)
        
        Args:
            model: Trained model with predict_proba method
            dataset: pandas DataFrame with features and target
            min_score: Minimum required Gini
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X)
            else:
                return TestResult(
                    passed=False,
                    failure_reason="Model must have predict_proba or decision_function method"
                )
            
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            gini = 2 * roc_auc - 1
            
            passed = gini >= min_score
            
            return TestResult(
                passed=passed,
                score=float(gini),
                details={
                    'gini': float(gini),
                    'roc_auc': float(roc_auc),
                    'min_score': min_score,
                    'num_samples': len(y_true)
                },
                failure_reason=f"Gini {gini:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing Gini: {str(e)}"
            )


@register_test
class ModelPrecision(ModelTest):
    """Test model precision"""
    
    name = "model.Precision"
    description = "Measure model precision"
    tags = ["performance", "classification"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.70, **config) -> TestResult:
        """
        Test model precision
        
        Args:
            model: Trained model
            dataset: pandas DataFrame
            min_score: Minimum required precision
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            y_pred = model.predict(X)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            
            passed = precision >= min_score
            
            return TestResult(
                passed=passed,
                score=float(precision),
                details={
                    'precision': float(precision),
                    'min_score': min_score
                },
                failure_reason=f"Precision {precision:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing precision: {str(e)}"
            )


@register_test
class ModelRecall(ModelTest):
    """Test model recall"""
    
    name = "model.Recall"
    description = "Measure model recall"
    tags = ["performance", "classification"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.70, **config) -> TestResult:
        """
        Test model recall
        
        Args:
            model: Trained model
            dataset: pandas DataFrame
            min_score: Minimum required recall
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            y_pred = model.predict(X)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            passed = recall >= min_score
            
            return TestResult(
                passed=passed,
                score=float(recall),
                details={
                    'recall': float(recall),
                    'min_score': min_score
                },
                failure_reason=f"Recall {recall:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing recall: {str(e)}"
            )


@register_test
class ModelF1Score(ModelTest):
    """Test model F1 score"""
    
    name = "model.F1Score"
    description = "Measure model F1 score"
    tags = ["performance", "classification"]
    
    def run(self, model: Any, dataset: Any, min_score: float = 0.70, **config) -> TestResult:
        """
        Test model F1 score
        
        Args:
            model: Trained model
            dataset: pandas DataFrame
            min_score: Minimum required F1
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]
        
        try:
            y_pred = model.predict(X)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            passed = f1 >= min_score
            
            return TestResult(
                passed=passed,
                score=float(f1),
                details={
                    'f1_score': float(f1),
                    'min_score': min_score
                },
                failure_reason=f"F1 score {f1:.3f} below threshold {min_score}" if not passed else None
            )
        except Exception as e:
            return TestResult(
                passed=False,
                failure_reason=f"Error computing F1 score: {str(e)}"
            )
