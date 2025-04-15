"""
This script analyzes datasets in benchmark directories, calculating statistics about 
data tables including row/column counts, data types, and missing values.

Usage:
    python profile_benchmark.py <input_dir> <output_dir>

Arguments:
    input_dir     Directory containing datalake/ and query/ folders
    output_dir    Directory for output files

Output:
    - JSON file with basic statistics about tables in the benchmark
"""

import argparse
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Regular expressions for type detection
RE_INTEGER = re.compile(r'^[-+]?\d+$')
RE_FLOAT = re.compile(r'^[-+]?\d+\.\d+$')


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (dict, defaultdict)):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(x) for x in obj]
    return obj


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


def collect_files(datalake_path: str, query_path: str) -> Tuple[List[str], List[str]]:
    """Collect all CSV files from both directories and handle duplicates"""
    # Get base filenames without path for easier comparison
    datalake_files = []
    datalake_basenames = set()
    query_files = []
    query_basenames = set()
    
    if os.path.exists(datalake_path):
        for f in os.listdir(datalake_path):
            if f.endswith('.csv'):
                datalake_files.append(os.path.join(datalake_path, f))
                datalake_basenames.add(f)
                
    if os.path.exists(query_path):
        for f in os.listdir(query_path):
            if f.endswith('.csv'):
                query_files.append(os.path.join(query_path, f))
                query_basenames.add(f)
                
    # Find duplicates and remove them from datalake files
    duplicates = datalake_basenames.intersection(query_basenames)
    filtered_datalake_files = [f for f in datalake_files 
                              if os.path.basename(f) not in duplicates]
    
    return filtered_datalake_files, query_files


def process_files(file_paths: List[str]) -> Dict:
    """Process files and return statistics with percentages"""
    stats = {
        "total_files": 0,
        "total_rows": 0,
        "total_columns": 0,
        "total_cells": 0,
        "missing_values": 0,
        "column_stats": defaultdict(int),
        "unique_columns": set(),
        "per_file_rows": [],
        "per_file_cols": []
    }
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
            stats["total_files"] += 1
            rows, cols = df.shape
            
            # Record dimensions
            stats["per_file_rows"].append(rows)
            stats["per_file_cols"].append(cols)
            stats["total_rows"] += rows
            stats["total_columns"] += cols
            stats["total_cells"] += rows * cols
            
            # Missing values
            missing = df.isna().sum().sum()
            stats["missing_values"] += missing
            
            # Column tracking
            for col in df.columns:
                stats["unique_columns"].add(col)
                dtype = determine_column_type(df[col])
                stats["column_stats"][dtype] += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Calculate derived metrics
    if stats["total_files"] > 0:
        stats["avg_rows"] = int(np.round(np.mean(stats["per_file_rows"])))
        stats["avg_cols"] = int(np.round(np.mean(stats["per_file_cols"])))
    else:
        stats["avg_rows"] = 0
        stats["avg_cols"] = 0
        
    # Add missing values percentage
    stats["missing_pct"] = round(
        (stats["missing_values"] / stats["total_cells"] * 100) 
        if stats["total_cells"] > 0 else 0, 2
    )
    
    # Add column type percentages
    total_cols = sum(stats["column_stats"].values())
    stats["column_pct"] = {
        k: round((v / total_cols * 100), 2) if total_cols > 0 else 0
        for k, v in stats["column_stats"].items()
    }
    
    # Cleanup and conversion
    stats["unique_columns"] = len(stats["unique_columns"])
    
    # Keep total counts but remove intermediate data
    del stats["per_file_rows"]
    del stats["per_file_cols"]
    
    return convert_numpy_types(stats)


def collect_unique_columns(file_paths: List[str]) -> Set:
    """Collect unique column names from a list of CSV files"""
    unique_cols = set()
    for f in file_paths:
        try:
            df = pd.read_csv(f)
            unique_cols.update(df.columns.astype(str))
        except Exception as e:
            print(f"Error reading columns from {f}: {str(e)}")
    return unique_cols


def profile_benchmark(input_dir: str) -> Dict:
    """Main profiling function with proper deduplication"""
    dl_path = os.path.join(input_dir, 'datalake')
    query_path = os.path.join(input_dir, 'query')
    
    # Validate directories
    if not os.path.exists(dl_path):
        raise ValueError(f"Missing datalake directory in {input_dir}")
    if not os.path.exists(query_path):
        raise ValueError(f"Missing query directory in {input_dir}")
        
    # Process both directories with deduplication
    dl_files, query_files = collect_files(dl_path, query_path)
    dl_stats = process_files(dl_files)
    query_stats = process_files(query_files)
    
    # Collect unique columns across both datasets
    dl_unique_cols = collect_unique_columns(dl_files)
    query_unique_cols = collect_unique_columns(query_files)
    combined_unique = dl_unique_cols.union(query_unique_cols)
    
    # Calculate combined statistics
    total_cells = dl_stats["total_cells"] + query_stats["total_cells"]
    combined_stats = {
        "total_files": dl_stats["total_files"] + query_stats["total_files"],
        "total_rows": dl_stats["total_rows"] + query_stats["total_rows"],
        "unique_columns": len(combined_unique),
        "missing_values": dl_stats["missing_values"] + query_stats["missing_values"],
        "total_columns": dl_stats["total_columns"] + query_stats["total_columns"],
        "total_cells": total_cells,
        "missing_pct": round(
            (dl_stats["missing_values"] + query_stats["missing_values"]) / total_cells * 100
            if total_cells > 0 else 0, 2
        )
    }
    
    return convert_numpy_types({
        "timestamp": datetime.now().isoformat(),
        "datalake": dl_stats,
        "query": query_stats,
        "combined": combined_stats
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile a data benchmarking directory')
    parser.add_argument('input_dir', help='Path to directory containing datalake/ and query/ folders')
    parser.add_argument('output_dir', help='Directory to save JSON profile')
    
    args = parser.parse_args()
    profile = profile_benchmark(args.input_dir)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.basename(os.path.normpath(args.input_dir))
    output_path = os.path.join(args.output_dir, f"{base_name}_overall.json")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2, default=str)
        
    print(f"Profile successfully saved to {output_path}")