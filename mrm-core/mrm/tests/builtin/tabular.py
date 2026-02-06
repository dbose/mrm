"""Built-in tests for tabular datasets"""

import numpy as np
import pandas as pd
from typing import Any
from mrm.tests.base import DatasetTest, TestResult
from mrm.tests.library import register_test


@register_test
class MissingValues(DatasetTest):
    """Check for missing values in dataset"""
    
    name = "tabular_dataset.MissingValues"
    description = "Detect and report missing values"
    tags = ["data_quality", "completeness"]
    
    def run(self, dataset: Any, threshold: float = 0.05, **config) -> TestResult:
        """
        Check for missing values
        
        Args:
            dataset: pandas DataFrame or similar
            threshold: Maximum allowed proportion of missing values
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        total_values = df.size
        missing_count = df.isnull().sum().sum()
        missing_ratio = missing_count / total_values if total_values > 0 else 0
        
        column_missing = df.isnull().sum() / len(df)
        columns_over_threshold = column_missing[column_missing > threshold].to_dict()
        
        passed = missing_ratio <= threshold
        
        return TestResult(
            passed=passed,
            score=1 - missing_ratio,
            details={
                'total_values': total_values,
                'missing_count': int(missing_count),
                'missing_ratio': float(missing_ratio),
                'threshold': threshold,
                'columns_over_threshold': {k: float(v) for k, v in columns_over_threshold.items()}
            },
            failure_reason=f"Missing value ratio {missing_ratio:.3f} exceeds threshold {threshold}" if not passed else None
        )


@register_test
class ClassImbalance(DatasetTest):
    """Check for class imbalance in target variable"""
    
    name = "tabular_dataset.ClassImbalance"
    description = "Detect severe class imbalance"
    tags = ["data_quality", "balance"]
    
    def run(self, dataset: Any, target_column: str = 'target', min_ratio: float = 0.1, **config) -> TestResult:
        """
        Check for class imbalance
        
        Args:
            dataset: pandas DataFrame
            target_column: Name of target column
            min_ratio: Minimum acceptable ratio for minority class
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        if target_column not in df.columns:
            return TestResult(
                passed=False,
                failure_reason=f"Target column '{target_column}' not found in dataset"
            )
        
        class_counts = df[target_column].value_counts()
        total = len(df)
        
        if len(class_counts) == 0:
            return TestResult(passed=False, failure_reason="No data in target column")
        
        min_count = class_counts.min()
        minority_ratio = min_count / total
        
        passed = minority_ratio >= min_ratio
        
        return TestResult(
            passed=passed,
            score=float(minority_ratio),
            details={
                'class_counts': class_counts.to_dict(),
                'minority_ratio': float(minority_ratio),
                'min_ratio': min_ratio,
                'total_samples': total
            },
            failure_reason=f"Minority class ratio {minority_ratio:.3f} below threshold {min_ratio}" if not passed else None
        )


@register_test
class OutlierDetection(DatasetTest):
    """Detect outliers using IQR method"""
    
    name = "tabular_dataset.OutlierDetection"
    description = "Detect outliers in numerical features"
    tags = ["data_quality", "outliers"]
    
    def run(self, dataset: Any, threshold: float = 0.1, iqr_multiplier: float = 1.5, **config) -> TestResult:
        """
        Detect outliers using IQR method
        
        Args:
            dataset: pandas DataFrame
            threshold: Maximum allowed proportion of outliers
            iqr_multiplier: IQR multiplier for outlier detection (default 1.5)
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_counts = {}
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = int(outliers)
            total_outliers += outliers
        
        total_values = len(df) * len(numeric_cols)
        outlier_ratio = total_outliers / total_values if total_values > 0 else 0
        
        passed = outlier_ratio <= threshold
        
        return TestResult(
            passed=passed,
            score=1 - outlier_ratio,
            details={
                'outlier_counts': outlier_counts,
                'total_outliers': total_outliers,
                'outlier_ratio': float(outlier_ratio),
                'threshold': threshold,
                'numeric_columns': list(numeric_cols)
            },
            failure_reason=f"Outlier ratio {outlier_ratio:.3f} exceeds threshold {threshold}" if not passed else None
        )


@register_test
class FeatureDistribution(DatasetTest):
    """Analyze feature distributions"""
    
    name = "tabular_dataset.FeatureDistribution"
    description = "Analyze statistical properties of features"
    tags = ["data_quality", "distribution"]
    
    def run(self, dataset: Any, **config) -> TestResult:
        """
        Analyze feature distributions
        
        Args:
            dataset: pandas DataFrame
        """
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            df = pd.DataFrame(dataset)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        return TestResult(
            passed=True,
            details={
                'distributions': distributions,
                'num_features': len(numeric_cols)
            }
        )
