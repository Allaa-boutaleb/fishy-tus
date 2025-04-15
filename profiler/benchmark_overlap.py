"""
This script analyzes the overlap between query tables and candidate tables in benchmarks.
It calculates column and value overlap metrics, and generates detailed statistics
about relationships between tables.

Usage:
    python benchmark_overlap.py <input_dir> <output_dir> [queries_file]

Arguments:
    input_dir      Directory containing datalake/, query/ folders and benchmark.pkl
    output_dir     Directory for output files
    queries_file   Optional file with list of specific query tables to analyze

Output:
    - summary JSON file with overall statistics
    - detailed JSON file with complete analysis results
"""

import argparse
import csv
import json
import os
import pickle
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

csv.field_size_limit(100000000)

import warnings
warnings.filterwarnings("ignore")

# Global cache to avoid reloading tables
TABLE_CACHE = {}
TOKEN_REGEX = re.compile(r'\b\w+\b')
# Regular expressions for type detection
RE_INTEGER = re.compile(r'^[-+]?\d+$')
RE_FLOAT = re.compile(r'^[-+]?\d+\.\d+$')

def main():
    """Main entry point for the benchmark analysis script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze overlap between query and candidate tables in benchmarks")
    parser.add_argument("input_dir", help="Input directory containing datalake/, query/ folders and benchmark.pkl")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("queries_file", nargs="?", help="Optional file with list of query tables to analyze")

    args = parser.parse_args()
    benchmark_name = os.path.basename(os.path.normpath(args.input_dir))

    # Get list of specific queries if provided
    specific_queries = None
    if args.queries_file:
        with open(args.queries_file, 'r') as f:
            specific_queries = [line.strip() for line in f if line.strip()]
        queries_basename = os.path.basename(args.queries_file)
        if '.' in queries_basename:
            queries_basename = queries_basename.split('.')[0]
        benchmark_name = queries_basename

    # Setup output files
    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = os.path.join(args.output_dir, f"{benchmark_name}_overlap_summary.json")
    detailed_file = os.path.join(args.output_dir, f"{benchmark_name}_overlap_detailed.json")

    # Run the benchmark analysis
    results = analyze_benchmark(
        input_dir=args.input_dir,
        specific_queries=specific_queries
    )

    summary_results = extract_summary(results)

    # Save results
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)

    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete.")
    print(f"Summary results saved to: {summary_file}")
    print(f"Detailed results saved to: {detailed_file}")

def extract_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract just the summary information from complete results."""
    summary = {
        "benchmark_name": results["benchmark_name"],
        "timestamp": results["timestamp"],
        "summary": results["summary"],
        "column_overlap_stats": results.get("column_overlap_stats", {}),
        "tuple_overlap_stats": results.get("tuple_overlap_stats", {}),
        "overlap_distribution": results.get("overlap_distribution", {}),
        # Include the new data type bin distribution
        "tuple_type_distribution": results.get("tuple_type_distribution", {})
    }
    return summary

def infer_value_type(value: Any) -> str:
    """Determine the data type of a single value using pattern matching"""
    if value is None or pd.isna(value):
        return "null"
    
    # Convert to string for pattern matching
    str_value = str(value).strip()
    if not str_value:
        return "null"
    
    # Check for different patterns
    if RE_INTEGER.match(str_value):
        return "integer"
    elif RE_FLOAT.match(str_value):
        return "float"
    
    # If it contains any letter, it's a string
    if any(c.isalpha() for c in str_value):
        return "string"
    
    # Fallback
    return "other"

def determine_column_type(series: pd.Series) -> str:
    """
    Improved type detection that examines column values for better type inference
    
    Args:
        series: Pandas Series to analyze
    
    Returns:
        String indicating the determined type (integer, float, string, or other)
    """
    # First use pandas built-in type detection for efficiency
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_string_dtype(series):
        return "string"
    
    # For categorical, datetime, and object types, examine actual values
    # Get non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return "other"  # Empty column
    
    # Sample values (use all if fewer than 100)
    sample_size = min(100, len(non_null))
    samples = non_null.sample(n=sample_size, random_state=42) if len(non_null) > sample_size else non_null
    
    # Count detected types
    type_counts = Counter(infer_value_type(val) for val in samples)
    
    # Determine dominant type
    if not type_counts:
        return "other"
    
    dominant_type, count = type_counts.most_common(1)[0]
    confidence = count / sum(type_counts.values())
    
    # If confidence is high enough, use the dominant type
    if confidence >= 0.7:  # 70% threshold
        if dominant_type == "null":
            return "other"
        return dominant_type
    
    # Handle special case: mixed numeric types (int and float)
    if set(type_counts.keys()).issubset({"integer", "float", "null"}):
        return "float"  # Use the more general type
    
    # Default to other for mixed types
    return "other"

def load_table(
    table_name: str,
    folder_path: str,
    fallback_folder_path: Optional[str] = None
) -> Optional[Tuple[pd.DataFrame, Dict[str, str], Dict[str, Set], str]]:
    """
    Load a table from CSV, detect data types, and extract values.
    Returns tuple of (DataFrame, column_types, values_by_type, location) or None if table not found.
    """
    # Check cache first
    if table_name in TABLE_CACHE:
        df, col_types, val_types = TABLE_CACHE[table_name]
        location = "unknown_cached"
        primary_path = os.path.join(folder_path, table_name)
        fallback_path = os.path.join(fallback_folder_path, table_name) if fallback_folder_path else None

        if os.path.exists(primary_path):
            location = os.path.basename(folder_path)
        elif fallback_path and os.path.exists(fallback_path):
            location = os.path.basename(fallback_folder_path)
        return df, col_types, val_types, location

    # Try to load from primary path
    file_path = os.path.join(folder_path, table_name)
    location = os.path.basename(folder_path)

    # Try fallback path if necessary
    if not os.path.exists(file_path) and fallback_folder_path:
        file_path = os.path.join(fallback_folder_path, table_name)
        location = os.path.basename(fallback_folder_path)

    if not os.path.exists(file_path):
        return None

    try:
        # Read CSV and process
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        col_types = {}
        
        # Filter out empty columns
        empty_columns = df.columns[df.isna().all()]
        if len(empty_columns) > 0:
            df = df.drop(columns=empty_columns)
            
        # For every column, get the type
        for col in df.columns:
            col_types[col] = determine_column_type(df[col])

        # Extract values by type
        val_types = extract_values_by_type(df, col_types)
        TABLE_CACHE[table_name] = (df, col_types, val_types)
        return df, col_types, val_types, location

    except Exception as e:
        print(f"Error loading or processing table {file_path}: {e}")
        return None

def analyze_benchmark(
    input_dir: str,
    specific_queries: Optional[List[str]] = None,
    exclude_self_comparisons: bool = True
) -> Dict[str, Any]:
    """
    Analyze relationships between query and candidate tables in a benchmark.
    
    Args:
        input_dir: Directory with datalake/, query/ folders and benchmark.pkl
        specific_queries: Optional list of specific query tables to analyze
        exclude_self_comparisons: Whether to exclude self-comparisons
        
    Returns:
        Dictionary with detailed analysis results
    """
    # Define paths and verify they exist
    datalake_folder = os.path.join(input_dir, "datalake")
    query_folder = os.path.join(input_dir, "query")
    benchmark_path = os.path.join(input_dir, "benchmark.pkl")

    if not os.path.exists(datalake_folder):
        raise FileNotFoundError(f"Datalake folder not found: {datalake_folder}")
    if not os.path.exists(query_folder):
        raise FileNotFoundError(f"Query folder not found: {query_folder}")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    # Load benchmark relationships
    with open(benchmark_path, 'rb') as f:
        benchmark_data = pickle.load(f)

    benchmark_name = os.path.basename(os.path.normpath(input_dir))

    # Initialize results structure
    results = {
        "benchmark_name": benchmark_name,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "query_relationships": {},
        "summary": {
            "total_queries": 0,
            "total_relationships": 0,
            "avg_candidates_per_query": 0,
            "excludes_self_comparisons": exclude_self_comparisons,
            "processed_queries": 0
        },
        "column_overlap_stats": {},
        "tuple_overlap_stats": {},
        "overlap_distribution": {
            "column_overlap_bins": defaultdict(lambda: {"count": 0, "percentage": 0.0}),
            "tuple_overlap_bins": defaultdict(lambda: {"count": 0, "percentage": 0.0})
        },
        # Add structure for tracking type-specific bin distributions
        "tuple_type_distribution": {
            data_type: {bin_name: {"count": 0, "percentage": 0.0} 
                       for bin_name in ["0-10", "10-20", "20-30", "30-40", "40-50", 
                                       "50-60", "60-70", "70-80", "80-90", "90-100"]}
            for data_type in ["integer", "float", "string", "other"]
        }
    }

    # Filter queries if specific ones are provided
    filtered_benchmark_keys = list(benchmark_data.keys())
    if specific_queries:
        valid_queries = set(specific_queries).intersection(set(filtered_benchmark_keys))
        filtered_benchmark_keys = [q for q in filtered_benchmark_keys if q in valid_queries]
        results["summary"]["requested_query_count"] = len(specific_queries)
        results["summary"]["found_query_count"] = len(valid_queries)
    else:
        results["summary"]["found_query_count"] = len(filtered_benchmark_keys)

    # Determine if queries can be candidates to themselves
    queries_as_candidates = False
    for query_name in filtered_benchmark_keys:
        if query_name in benchmark_data.get(query_name, []):
            queries_as_candidates = True
            break
    results["summary"]["queries_as_candidates"] = queries_as_candidates

    # Collect all query-candidate pairs for processing
    work_items = []
    for query_name in filtered_benchmark_keys:
        candidates = benchmark_data.get(query_name, [])
        filtered_candidates = [c for c in candidates if not (exclude_self_comparisons and c == query_name)]
        for candidate_name in filtered_candidates:
            work_items.append((query_name, candidate_name))

    # Track statistics 
    all_column_overlaps = []
    all_tuple_overlaps = []
    all_tuple_overlaps_by_type = defaultdict(list)
    
    # Track type-specific counts per bin
    type_bin_counts = defaultdict(lambda: defaultdict(int))
    
    # Initialize per-query data
    query_data = {query_name: {
        "total_candidate_count": len(benchmark_data.get(query_name, [])),
        "analyzed_candidate_count": 0,
        "candidates": {},
        "column_overlaps": [],
        "tuple_overlaps": [],
        "tuple_overlaps_by_type": defaultdict(list),
    } for query_name in filtered_benchmark_keys}

    # Process all query-candidate pairs
    processed_query_count = 0
    processed_relationships_count = 0
    
    for query_name, candidate_name in tqdm(work_items, desc="Processing"):
        if exclude_self_comparisons and query_name == candidate_name:
            continue
            
        # Load tables and skip if either can't be loaded
        query_processed_data = load_table(query_name, query_folder)
        if query_processed_data is None:
            continue
            
        candidate_processed_data = load_table(candidate_name, datalake_folder, query_folder)
        if candidate_processed_data is None:
            continue
            
        query_df, query_col_types, query_values_by_type, _ = query_processed_data
        candidate_df, candidate_col_types, candidate_values_by_type, candidate_location = candidate_processed_data
        
        # Calculate overlap metrics - only consider column name overlap, not type
        column_overlap = calculate_column_overlap(query_df, candidate_df)
        tuple_overlap = calculate_tuple_overlap(query_values_by_type, candidate_values_by_type)
        
        # Store results for this pair
        result = {
            "column_overlap": column_overlap,
            "tuple_overlap": tuple_overlap,
            "location": candidate_location,
            "is_self": (query_name == candidate_name)
        }
            
        # Update query data with results
        query_data[query_name]["analyzed_candidate_count"] += 1
        query_data[query_name]["candidates"][candidate_name] = result
        processed_relationships_count += 1
        
        # Extract and track key metrics
        col_ov_coeff = result["column_overlap"]["overlap_coefficient"]
        tup_ov_coeff = result["tuple_overlap"]["overall"]["overlap_coefficient"]
        
        all_column_overlaps.append(col_ov_coeff)
        all_tuple_overlaps.append(tup_ov_coeff)
        
        query_data[query_name]["column_overlaps"].append(col_ov_coeff)
        query_data[query_name]["tuple_overlaps"].append(tup_ov_coeff)
        
        # Update distribution bins
        col_bin = determine_bin(col_ov_coeff * 100)
        tuple_bin = determine_bin(tup_ov_coeff * 100)
        results["overlap_distribution"]["column_overlap_bins"][col_bin]["count"] += 1
        results["overlap_distribution"]["tuple_overlap_bins"][tuple_bin]["count"] += 1
        
        # Track type-specific statistics and bin distributions
        for data_type, overlap in result["tuple_overlap"].get("by_type", {}).items():
            ov_coeff = overlap.get("overlap_coefficient")
            if ov_coeff is not None:
                all_tuple_overlaps_by_type[data_type].append(ov_coeff)
                query_data[query_name]["tuple_overlaps_by_type"][data_type].append(ov_coeff)
                
                # Track type-specific bin counts
                type_bin = determine_bin(ov_coeff * 100)
                type_bin_counts[data_type][type_bin] += 1
    
    # Post-process query data and add to results
    for query_name, qdata in query_data.items():
        if qdata["analyzed_candidate_count"] > 0:
            processed_query_count += 1
            
            # Calculate query-specific statistics
            if qdata["column_overlaps"]:
                qdata["column_overlap_stats"] = calculate_statistics(qdata["column_overlaps"])
            
            if qdata["tuple_overlaps"]:
                qdata["tuple_overlap_stats"] = {
                    "overall": calculate_statistics(qdata["tuple_overlaps"]),
                    "by_type": {
                        data_type: calculate_statistics(overlaps)
                        for data_type, overlaps in qdata["tuple_overlaps_by_type"].items() if overlaps
                    }
                }
            
            # Clean up temporary data before adding to final results
            del qdata["column_overlaps"]
            del qdata["tuple_overlaps"]
            del qdata["tuple_overlaps_by_type"]
            
            # Add to final results
            results["query_relationships"][query_name] = qdata

    # Calculate overall summary statistics
    results["summary"]["total_queries"] = processed_query_count
    results["summary"]["total_relationships"] = processed_relationships_count
    results["summary"]["processed_queries"] = processed_query_count

    if processed_query_count > 0:
        results["summary"]["avg_candidates_per_query"] = processed_relationships_count / processed_query_count

    # Calculate detailed statistics for column and tuple overlap
    if all_column_overlaps:
        results["column_overlap_stats"] = calculate_statistics(all_column_overlaps)
        
    if all_tuple_overlaps:
        results["tuple_overlap_stats"] = {
            "overall": calculate_statistics(all_tuple_overlaps),
            "by_type": {
                data_type: calculate_statistics(overlaps)
                for data_type, overlaps in all_tuple_overlaps_by_type.items() if overlaps
            }
        }

    # Convert bin counts to percentages
    if processed_relationships_count > 0:
        for bin_name, data in results["overlap_distribution"]["column_overlap_bins"].items():
            data["percentage"] = (data["count"] / processed_relationships_count) * 100
        for bin_name, data in results["overlap_distribution"]["tuple_overlap_bins"].items():
            data["percentage"] = (data["count"] / processed_relationships_count) * 100
            
        # Process type-specific bin distributions
        for data_type, bin_counts in type_bin_counts.items():
            type_total = sum(bin_counts.values())
            if type_total > 0:
                for bin_name, count in bin_counts.items():
                    results["tuple_type_distribution"][data_type][bin_name] = {
                        "count": count,
                        "percentage": (count / type_total) * 100
                    }
    else:
        # Ensure bins exist even if empty
        bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
        for bin_name in bins:
            results["overlap_distribution"]["column_overlap_bins"].setdefault(bin_name, {"count": 0, "percentage": 0.0})
            results["overlap_distribution"]["tuple_overlap_bins"].setdefault(bin_name, {"count": 0, "percentage": 0.0})

    # Count self-comparisons
    self_comparison_count = 0
    source_keys = specific_queries if specific_queries else benchmark_data.keys()
    for query_name in source_keys:
        if query_name in benchmark_data.get(query_name, []):
            if specific_queries and query_name in filtered_benchmark_keys:
                self_comparison_count += 1
            elif not specific_queries:
                self_comparison_count += 1

    results["summary"]["self_comparison_count"] = self_comparison_count
    TABLE_CACHE.clear()

    return results

def determine_bin(overlap_pct: float) -> str:
    """Determine which 10% bin a given overlap percentage falls into."""
    if overlap_pct < 0: overlap_pct = 0
    if overlap_pct > 100: overlap_pct = 100
    if overlap_pct == 100:
        bin_idx = 9
    else:
        bin_idx = int(overlap_pct // 10)

    bins = ["0-10", "10-20", "20-30", "30-40", "40-50",
            "50-60", "60-70", "70-80", "80-90", "90-100"]
    return bins[bin_idx]

def tokenize_string(text: Any) -> Set[str]:
    """Extract individual word tokens from a string."""
    if pd.isna(text):
        return set()
    if not isinstance(text, str):
        text = str(text)

    tokens = set(TOKEN_REGEX.findall(text.lower()))
    return tokens

def extract_values_by_type(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Set]:
    """
    Extract all unique values from a DataFrame, organized by data type.
    For string columns, tokenizes into individual words.
    """
    values_by_type = defaultdict(set)

    for col, data_type in column_types.items():
        if col not in df.columns:
            continue

        col_series = df[col].dropna()

        if col_series.empty:
            continue

        # Process strings by tokenizing
        if data_type == "string":
            try:
                # Vectorized string tokenization
                string_series = col_series.astype(str)
                token_lists = string_series.str.lower().str.findall(TOKEN_REGEX)
                all_tokens = token_lists.explode().dropna()
                unique_tokens = set(all_tokens.unique())
                values_by_type[data_type].update(unique_tokens)
            except Exception as e:
                # Fallback to row-by-row processing
                for val in col_series:
                    values_by_type[data_type].update(tokenize_string(val))

        # Handle numeric data types
        elif data_type == "integer":
            try:
                unique_vals = set(col_series.astype(int).unique())
                values_by_type[data_type].update(unique_vals)
            except (TypeError, ValueError):
                # If conversion fails, treat as string
                unique_vals = set(col_series.astype(str).unique())
                values_by_type["string"].update(tokenize_string(" ".join(map(str, unique_vals))))
                
        elif data_type == "float":
            try:
                unique_vals = set(np.round(col_series.astype(float), decimals=8).unique())
                values_by_type[data_type].update(unique_vals)
            except (TypeError, ValueError):
                # If conversion fails, treat as string
                unique_vals = set(col_series.astype(str).unique())
                values_by_type["string"].update(tokenize_string(" ".join(map(str, unique_vals))))
                
        # Everything else goes to other
        else:
            try:
                # For "other" type, try to cast to string and tokenize
                unique_vals = set(col_series.astype(str).unique())
                values_by_type["other"].update(unique_vals)
            except Exception:
                pass

    return dict(values_by_type)

def calculate_column_overlap(
    query_df: pd.DataFrame,
    candidate_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate column name overlap metrics between query and candidate tables.
    Uses Overlap Coefficient and Jaccard Similarity.
    """
    # Get column sets
    query_columns = set(query_df.columns)
    candidate_columns = set(candidate_df.columns)

    # Calculate overall overlap
    overlap_columns = query_columns.intersection(candidate_columns)
    overlap_count = len(overlap_columns)

    # Calculate Overlap Coefficient (Szymkiewicz–Simpson coefficient)
    min_size = min(len(query_columns), len(candidate_columns))
    overlap_coefficient = overlap_count / min_size if min_size > 0 else 0.0

    # Calculate Jaccard similarity for reference
    union_columns = query_columns.union(candidate_columns)
    jaccard_similarity = (overlap_count / len(union_columns)) if union_columns else 0.0

    return {
        "query_column_count": len(query_columns),
        "candidate_column_count": len(candidate_columns),
        "overlap_count": overlap_count,
        "overlap_coefficient": overlap_coefficient,
        "jaccard_similarity": jaccard_similarity,
        "overlap_columns": list(overlap_columns),
    }

def calculate_tuple_overlap(
    query_values_by_type: Dict[str, Set],
    candidate_values_by_type: Dict[str, Set]
) -> Dict[str, Any]:
    """
    Calculate tuple (value) overlap metrics using Overlap Coefficient.
    Analyzes both overall overlap and overlap by data type.
    """
    # Combine all values across types for overall calculation
    query_values_all = set().union(*query_values_by_type.values()) if query_values_by_type else set()
    candidate_values_all = set().union(*candidate_values_by_type.values()) if candidate_values_by_type else set()

    # Calculate overall overlap
    common_values = query_values_all & candidate_values_all
    common_count = len(common_values)

    # Calculate metrics
    min_size = min(len(query_values_all), len(candidate_values_all))
    overlap_coefficient = common_count / min_size if min_size > 0 else 0.0

    union_values = query_values_all | candidate_values_all
    jaccard_similarity = common_count / len(union_values) if union_values else 0.0

    # Calculate overlap by type
    overlap_by_type = {}
    all_types = set(query_values_by_type.keys()) | set(candidate_values_by_type.keys())

    for data_type in all_types:
        query_type_values = query_values_by_type.get(data_type, set())
        candidate_type_values = candidate_values_by_type.get(data_type, set())

        if not query_type_values and not candidate_type_values:
            continue

        type_common_values = query_type_values & candidate_type_values
        type_common_count = len(type_common_values)

        type_min_size = min(len(query_type_values), len(candidate_type_values))
        type_overlap_coef = type_common_count / type_min_size if type_min_size > 0 else 0.0

        type_union = query_type_values | candidate_type_values
        type_jaccard = type_common_count / len(type_union) if type_union else 0.0

        overlap_by_type[data_type] = {
            "query_value_count": len(query_type_values),
            "candidate_value_count": len(candidate_type_values),
            "common_value_count": type_common_count,
            "overlap_coefficient": type_overlap_coef,
            "jaccard_similarity": type_jaccard
        }

    return {
        "overall": {
            "query_value_count": len(query_values_all),
            "candidate_value_count": len(candidate_values_all),
            "common_value_count": common_count,
            "overlap_coefficient": overlap_coefficient,
            "jaccard_similarity": jaccard_similarity
        },
        "by_type": overlap_by_type
    }

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {
            "count": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
            "min": np.nan, "max": np.nan, "q1": np.nan, "q3": np.nan,
            "iqr": np.nan, "p10": np.nan, "p90": np.nan
        }

    data = np.array(values, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {
            "count": len(values),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "p10": float(np.percentile(data, 10)),
            "p90": float(np.percentile(data, 90))
        }

    # Fix NaN results if appropriate
    for key, value in stats.items():
        if np.isnan(value) and key != 'count':
            if key == 'std' and len(set(values)) == 1:
                stats[key] = 0.0

    return stats

if __name__ == "__main__":
    main()