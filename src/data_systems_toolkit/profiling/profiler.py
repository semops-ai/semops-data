"""
Data profiling capabilities using ysemops-dataofiling and custom analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings("ignore")

from ..core.logging import get_logger

logger = get_logger(__name__)


class DataProfiler:
 """Data profiling and quality assessment toolkit."""

 def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
 """Initialize data profiler.

 Args:
 config_overrides: Custom configuration for ProfileReport
 """
 self.default_config = {
 "title": "Dataset Profile Report",
 "explorative": True,
 "minimal": False,
 }

 if config_overrides:
 self.default_config.update(config_overrides)

 def profile_dataframe(
 self, df: pd.DataFrame, title: str = "DataFrame Profile", minimal: bool = False
 ) -> ProfileReport:
 """Profile a pandas DataFrame.

 Args:
 df: DataFrame to profile
 title: Title for the report
 minimal: Use minimal mode for faster profiling

 Returns:
 ProfileReport: ysemops-dataofiling report object
 """
 logger.info(f"Profiling DataFrame with {len(df)} rows, {len(df.columns)} columns")

 config = self.default_config.copy
 config["title"] = title
 config["minimal"] = minimal

 profile = ProfileReport(df, **config)
 return profile

 def profile_file(
 self,
 file_path: Union[str, Path],
 title: Optional[str] = None,
 sample_size: Optional[int] = None,
 ) -> ProfileReport:
 """Profile a data file (CSV, Excel, Parquet, etc.).

 Args:
 file_path: Path to the data file
 title: Custom title for the report
 sample_size: Number of rows to sample (None for all data)

 Returns:
 ProfileReport: ysemops-dataofiling report object
 """
 logger.info(f"Profiling file: {file_path}")

 df = self._load_file(file_path, sample_size)

 config = self.default_config.copy
 if title:
 config["title"] = title
 else:
 config["title"] = f"Profile: {Path(file_path).name}"

 profile = ProfileReport(df, **config)
 logger.info(f"Generated profile for {len(df)} rows, {len(df.columns)} columns")

 return profile

 def quick_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Generate quick summary statistics.

 Args:
 df: DataFrame to summarize

 Returns:
 Dict containing summary statistics
 """
 summary = {
 "basic_info": {
 "shape": df.shape,
 "memory_usage_mb": df.memory_usage(deep=True).sum / 1024**2,
 "dtypes_count": df.dtypes.value_counts.to_dict,
 },
 "missing_data": {
 "total_missing": df.isnull.sum.sum,
 "missing_percentage": (df.isnull.sum.sum / (df.shape[0] * df.shape[1]))
 * 100,
 "columns_with_missing": df.columns[df.isnull.any].tolist,
 },
 "duplicates": {
 "duplicate_rows": df.duplicated.sum,
 "duplicate_percentage": (df.duplicated.sum / len(df)) * 100,
 },
 }

 # Numeric columns analysis
 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist
 if numeric_cols:
 summary["numeric_analysis"] = {
 "numeric_columns_count": len(numeric_cols),
 "numeric_columns": numeric_cols,
 "correlation_high": self._find_high_correlations(df[numeric_cols]),
 }

 # Categorical columns analysis
 categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist
 if categorical_cols:
 summary["categorical_analysis"] = {
 "categorical_columns_count": len(categorical_cols),
 "categorical_columns": categorical_cols,
 "high_cardinality_columns": [
 col for col in categorical_cols if df[col].nunique > 50
 ],
 }

 # Date columns analysis
 date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist
 if date_cols:
 summary["temporal_analysis"] = {
 "date_columns_count": len(date_cols),
 "date_columns": date_cols,
 "date_ranges": {
 col: {
 "min": df[col].min,
 "max": df[col].max,
 "range_days": (df[col].max - df[col].min).days,
 }
 for col in date_cols
 },
 }

 return summary

 def data_quality_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Assess data quality issues.

 Args:
 df: DataFrame to assess

 Returns:
 Dict containing quality assessment
 """
 quality_issues = {
 "completeness": self._assess_completeness(df),
 "consistency": self._assess_consistency(df),
 "validity": self._assess_validity(df),
 "uniqueness": self._assess_uniqueness(df),
 }

 # Calculate overall quality score
 quality_scores = []
 for category, issues in quality_issues.items:
 if "score" in issues:
 quality_scores.append(issues["score"])

 quality_issues["overall_score"] = np.mean(quality_scores) if quality_scores else 0.0

 return quality_issues

 def compare_datasets(
 self,
 df1: pd.DataFrame,
 df2: pd.DataFrame,
 df1_name: str = "Dataset 1",
 df2_name: str = "Dataset 2",
 ) -> Dict[str, Any]:
 """Compare two datasets for schema and distribution differences.

 Args:
 df1: First dataset
 df2: Second dataset
 df1_name: Name for first dataset
 df2_name: Name for second dataset

 Returns:
 Dict containing comparison results
 """
 comparison = {
 "schema_comparison": self._compare_schemas(df1, df2, df1_name, df2_name),
 "distribution_comparison": self._compare_distributions(
 df1, df2, df1_name, df2_name
 ),
 "summary": {},
 }

 comparison["summary"] = {
 "schema_compatible": len(comparison["schema_comparison"]["column_differences"])
 == 0,
 "distribution_drift_detected": len(
 comparison["distribution_comparison"]["significant_differences"]
 )
 > 0,
 "recommendation": self._generate_comparison_recommendation(comparison),
 }

 return comparison

 def _load_file(
 self, file_path: Union[str, Path], sample_size: Optional[int] = None
 ) -> pd.DataFrame:
 """Load data file into DataFrame."""
 file_path = Path(file_path)

 if not file_path.exists:
 raise FileNotFoundError(f"File not found: {file_path}")

 if file_path.suffix.lower == ".csv":
 df = pd.read_csv(file_path, nrows=sample_size)
 elif file_path.suffix.lower in [".xlsx", ".xls"]:
 df = pd.read_excel(file_path, nrows=sample_size)
 elif file_path.suffix.lower == ".parquet":
 df = pd.read_parquet(file_path)
 if sample_size:
 df = df.head(sample_size)
 elif file_path.suffix.lower == ".json":
 df = pd.read_json(file_path)
 if sample_size:
 df = df.head(sample_size)
 else:
 raise ValueError(f"Unsupported file format: {file_path.suffix}")

 return df

 def _find_high_correlations(
 self, df: pd.DataFrame, threshold: float = 0.8
 ) -> List[Tuple[str, str, float]]:
 """Find highly correlated numeric columns."""
 if len(df.columns) < 2:
 return []

 corr_matrix = df.corr
 high_corr_pairs = []

 for i in range(len(corr_matrix.columns)):
 for j in range(i + 1, len(corr_matrix.columns)):
 corr_value = corr_matrix.iloc[i, j]
 if abs(corr_value) >= threshold:
 high_corr_pairs.append(
 (
 corr_matrix.columns[i],
 corr_matrix.columns[j],
 corr_value,
 )
 )

 return high_corr_pairs

 def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Assess data completeness."""
 missing_stats = df.isnull.sum
 total_cells = len(df) * len(df.columns)
 missing_cells = missing_stats.sum

 return {
 "score": 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0,
 "missing_percentage": (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
 "columns_missing_data": missing_stats[missing_stats > 0].to_dict,
 "completely_missing_columns": missing_stats[
 missing_stats == len(df)
 ].index.tolist,
 }

 def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Assess data consistency."""
 issues = []

 for col in df.select_dtypes(include=["object"]):
 try:
 pd.to_numeric(df[col].dropna)
 issues.append(f"Column '{col}' contains mixed data types")
 except (ValueError, TypeError):
 pass

 return {
 "score": 1.0 - (len(issues) / len(df.columns)) if len(df.columns) > 0 else 1.0,
 "issues": issues,
 }

 def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Assess data validity."""
 validity_issues = []

 for col in df.select_dtypes(include=[np.number]):
 if (df[col] < 0).any and col.lower in [
 "age",
 "price",
 "count",
 "quantity",
 ]:
 validity_issues.append(f"Negative values in {col} (potentially invalid)")

 return {
 "score": 1.0 - (len(validity_issues) / len(df.columns)) if len(df.columns) > 0 else 1.0,
 "issues": validity_issues,
 }

 def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
 """Assess data uniqueness."""
 duplicate_rows = df.duplicated.sum

 return {
 "score": 1.0 - (duplicate_rows / len(df)) if len(df) > 0 else 1.0,
 "duplicate_rows": duplicate_rows,
 "duplicate_percentage": (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
 }

 def _compare_schemas(
 self, df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str
 ) -> Dict[str, Any]:
 """Compare schemas of two datasets."""
 cols1, cols2 = set(df1.columns), set(df2.columns)
 types1, types2 = df1.dtypes.to_dict, df2.dtypes.to_dict

 return {
 "columns_only_in_" + name1.lower.replace(" ", "_"): list(cols1 - cols2),
 "columns_only_in_" + name2.lower.replace(" ", "_"): list(cols2 - cols1),
 "common_columns": list(cols1 & cols2),
 "column_differences": [
 col for col in (cols1 & cols2) if str(types1[col]) != str(types2[col])
 ],
 }

 def _compare_distributions(
 self, df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str
 ) -> Dict[str, Any]:
 """Compare distributions of common numeric columns."""
 common_cols = list(set(df1.columns) & set(df2.columns))
 numeric_cols = [
 col for col in common_cols if df1[col].dtype in ["int64", "float64"]
 ]

 differences = []
 for col in numeric_cols:
 stats1 = df1[col].describe
 stats2 = df2[col].describe

 if stats1["mean"] != 0:
 mean_diff_pct = abs(stats1["mean"] - stats2["mean"]) / abs(stats1["mean"]) * 100
 if mean_diff_pct > 20:
 differences.append(
 {
 "column": col,
 "metric": "mean",
 "difference_pct": mean_diff_pct,
 name1: stats1["mean"],
 name2: stats2["mean"],
 }
 )

 return {
 "compared_columns": numeric_cols,
 "significant_differences": differences,
 }

 def _generate_comparison_recommendation(self, comparison: Dict[str, Any]) -> str:
 """Generate recommendation based on comparison results."""
 if not comparison["summary"]["schema_compatible"]:
 return "Schema differences detected. Consider data alignment before further analysis."
 elif comparison["summary"]["distribution_drift_detected"]:
 return "Distribution drift detected. Investigate potential data quality issues or temporal changes."
 else:
 return "Datasets appear compatible for combined analysis."
