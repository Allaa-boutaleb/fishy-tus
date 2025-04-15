# run.py

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import argparse
import re
import hashlib

# Import embedding extraction classes
try:
    from baselines.hash_extractor import HashEmbeddingExtractor
    from baselines.sbert_extractor import SbertEmbeddingExtractor
    from baselines.tfidf_extractor import TfidfEmbeddingExtractor
    from baselines.count_extractor import CountEmbeddingExtractor
except ImportError as e:
    print(f"Error importing embedding extractors: {e}", file=sys.stderr)
    print("Please ensure the 'baselines' package is correctly installed and accessible.", file=sys.stderr)
    sys.exit(1)

# Import the evaluation function
try:
    # This assumes evaluate_benchmark_embeddings returns
    # a dict like: {'metrics': {...}, 'detailed_results': {...}}
    from evaluation import evaluate_benchmark_embeddings
except ImportError as e:
    print(f"Error importing evaluation function: {e}", file=sys.stderr)
    print("Please ensure 'evaluation.py' is accessible and modified to return detailed results.", file=sys.stderr)
    sys.exit(1)

# Benchmark-specific configurations
BENCHMARK_CONFIGS = {
    "santos": {"exclude_self_matches": False},
    "tus": {"exclude_self_matches": False, "min_columns": 10, "exclude_roots": ["t_28dc8f7610402ea7"]},
    "tusLarge": {"exclude_self_matches": False, "min_columns": 10, "exclude_roots": ["t_28dc8f7610402ea7"]},
    "pylon": {"exclude_self_matches": False}
}

# METHOD_GRIDS specification
METHOD_GRIDS = {
    "hash": {
        "embedding_params": {
            "sample_size": [1000],
            "n_features": [4096],
            "include_column_names": [False]
        },
        "eval_params": {
            "agg": ["max"],
            "threshold": []  # Only used when agg="None"
        }
    },
    "count": {
        "embedding_params": {
            "sample_size": [1000],
            "max_features": [4096],
            "ngram_range": [(1, 2)],
            "include_column_names": [False]
        },
        "eval_params": {
            "agg": ["max"],
            "threshold": []  # Only used when agg="None"
        }
    },
    "tfidf": {
        "embedding_params": {
            "sample_size": [1000],
            "max_features": [4096],
            "ngram_range": [(1, 2)],
            "include_column_names": [False]
        },
        "eval_params": {
            "agg": ["max"],
            "threshold": []  # Only used when agg="None"
        }
    },
    "sbert": {
        "embedding_params": {
            "sample_size": [20],
            "model_name": ["all-mpnet-base-v2"],
            "orientation": ["vertical"],
            "deduplicate": [True]
            # include_names and names_only handled separately
        },
        "eval_params": {
            "agg": ["mean"],
            "threshold": [0.1]  # Only used when agg="None"
        }
    }
}

# Cache for vectorizers to ensure consistency between runs
VECTORIZER_CACHE = {}

# --- Helper Functions ---

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def get_query_files(benchmark_name):
    """Gets all CSV files from the benchmark's query directory."""
    query_dir = f'data/{benchmark_name}/query'
    if not os.path.isdir(query_dir):
        print(f"Error: Query directory not found at {query_dir}", file=sys.stderr)
        sys.exit(1)
    all_query_files = [f for f in os.listdir(query_dir) if f.endswith('.csv')]
    print(f"Found {len(all_query_files)} query files in {query_dir}")
    return all_query_files

def load_ground_truth(benchmark_name):
    """Loads the ground truth pickle file for the benchmark."""
    gt_path = f'data/{benchmark_name}/benchmark.pkl'
    try:
        with open(gt_path, 'rb') as f:
            ground_truth = pickle.load(f)
            print(f"Loaded ground truth for {benchmark_name} with {len(ground_truth)} queries.")
            return ground_truth
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading ground truth file {gt_path}: {e}", file=sys.stderr)
        sys.exit(1)

def generate_filename(benchmark, embed_type, embed_params, eval_params):
    """Generates a descriptive filename based on parameters."""
    embed_items = sorted(embed_params.items())
    eval_items = sorted(eval_params.items())

    # Create string representation, ensuring consistent order
    param_str_parts = [embed_type]
    param_str_parts.extend([f"{k}={v}" for k, v in embed_items])
    param_str_parts.extend([f"{k}={v}" for k, v in eval_items])
    param_str = "_".join(param_str_parts)

    # Clean the string for use in filename
    param_str = re.sub(r'[^\w\-_\.]', '_', param_str)  # Allow word chars, hyphen, underscore, dot
    param_str = re.sub(r'_+', '_', param_str).strip('_')  # Replace multiple underscores, strip ends

    # Use a hash if the filename gets too long or contains problematic sequences
    max_len = 200  # Max filename length (adjust as needed)
    if len(param_str) > max_len:
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        # Keep method type for easier identification
        param_str = f"{embed_type}_{param_hash}"

    return f"{benchmark}_{param_str}.json"

def parameter_combinations(grid):
    """Generates all combinations of parameters from a grid."""
    if not grid:
        return [{}]
    keys = grid.keys()
    values = grid.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

# --- Core Logic ---

def create_extractor(embedding_type, params):
    """Creates an embedding extractor based on the type and parameters."""
    if embedding_type == "hash":
        return HashEmbeddingExtractor(
            sample_size=params.get("sample_size", 5),
            n_features=params.get("n_features", 1024),
            include_column_names=params.get("include_column_names", False)
        )
    elif embedding_type == "count":
        return CountEmbeddingExtractor(
            sample_size=params.get("sample_size", 5),
            max_features=params.get("max_features", 1024),
            ngram_range=params.get("ngram_range", (1, 1)),
            include_column_names=params.get("include_column_names", True)
        )
    elif embedding_type == "tfidf":
        return TfidfEmbeddingExtractor(
            sample_size=params.get("sample_size", 5),
            max_features=params.get("max_features", 1024),
            ngram_range=params.get("ngram_range", (1, 2)),
            include_column_names=params.get("include_column_names", True)
        )
    elif embedding_type == "sbert":
        return SbertEmbeddingExtractor(
            sample_size=params.get("sample_size", 10),
            model_name=params.get("model_name", "all-mpnet-base-v2"),
            orientation=params.get("orientation", "vertical"),
            include_names=params.get("include_names", True),
            names_only=params.get("names_only", False),
            deduplicate=params.get("deduplicate", True),
            batch_size=params.get("batch_size", 256)
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

def generate_embeddings(benchmark_name, embedding_type, params, query_files=None):
    """
    Generates or loads embeddings for a benchmark using specified method and parameters.
    Returns datalake embeddings, query embeddings, generation time, and file paths.
    """
    datalake_dir = f'data/{benchmark_name}/datalake'
    query_dir = f'data/{benchmark_name}/query'
    vector_dir = f'vectors/{benchmark_name}'
    ensure_dir(vector_dir)

    param_tuple = tuple(sorted(params.items()))
    param_hash = hashlib.md5(str(param_tuple).encode()).hexdigest()
    datalake_output = os.path.join(vector_dir, f'dl_{embedding_type}_{param_hash}.pkl')
    query_output = os.path.join(vector_dir, f'q_{embedding_type}_{param_hash}.pkl')

    if os.path.exists(datalake_output) and os.path.exists(query_output):
        print(f"Loading existing embeddings for {embedding_type} (hash: {param_hash})")
        try:
            with open(datalake_output, 'rb') as f: 
                dl_data = pickle.load(f)
            with open(query_output, 'rb') as f: 
                q_data = pickle.load(f)
            datalake_embeddings = dl_data[0] if embedding_type == "sbert" else dl_data
            query_embeddings = q_data[0] if embedding_type == "sbert" else q_data
            return datalake_embeddings, query_embeddings, 0, datalake_output, query_output
        except Exception as e:
            print(f"Warning: Error loading existing embeddings: {e}. Regenerating...")

    start_time = time.time()
    print(f"Generating embeddings for {embedding_type} with params: {params}")

    # Use a vectorizer cache key for consistent processing
    vectorizer_cache_key = (embedding_type,) + param_tuple
    
    try:
        if embedding_type in ["count", "tfidf"]:
            # Reuse cached vectorizer if available
            extractor = create_extractor(embedding_type, params)
            vectorizer = VECTORIZER_CACHE.get(vectorizer_cache_key)
            
            # Process datalake first
            vectorizer = extractor.process_directory(
                input_dir=datalake_dir,
                output_path=datalake_output,
                vectorizer=vectorizer
            )
            
            # Cache the vectorizer
            VECTORIZER_CACHE[vectorizer_cache_key] = vectorizer
            
            # Process queries with the same vectorizer
            extractor.process_directory(
                input_dir=query_dir,
                output_path=query_output,
                selected_files=query_files,
                vectorizer=vectorizer  # Critical: Use the same vectorizer
            )
            
        elif embedding_type == "hash":
            extractor = create_extractor(embedding_type, params)
            extractor.process_directory(
                input_dir=datalake_dir,
                output_path=datalake_output
            )
            extractor.process_directory(
                input_dir=query_dir,
                output_path=query_output,
                selected_files=query_files
            )
            
        elif embedding_type == "sbert":
            extractor = create_extractor(embedding_type, params)
            # SBERT is stateful, so the model is cached internally
            extractor.process_directory(
                input_dir=datalake_dir,
                output_path=datalake_output
            )
            extractor.process_directory(
                input_dir=query_dir,
                output_path=query_output,
                selected_files=query_files
            )
            
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
            
    except Exception as e:
        print(f"Error during embedding generation for {embedding_type} with params {params}: {e}", file=sys.stderr)
        if os.path.exists(datalake_output): 
            os.remove(datalake_output)
        if os.path.exists(query_output): 
            os.remove(query_output)
        raise

    generation_time = time.time() - start_time
    print(f"Generation finished in {generation_time:.2f} seconds.")

    try:
        with open(datalake_output, 'rb') as f: 
            dl_data = pickle.load(f)
        with open(query_output, 'rb') as f: 
            q_data = pickle.load(f)
        datalake_embeddings = dl_data[0] if embedding_type == "sbert" else dl_data
        query_embeddings = q_data[0] if embedding_type == "sbert" else q_data
        return datalake_embeddings, query_embeddings, generation_time, datalake_output, query_output
    except Exception as e:
        print(f"Error loading newly generated embeddings: {e}", file=sys.stderr)
        raise

def evaluate_single_combination(datalake_embeddings, query_embeddings, ground_truth, eval_params, benchmark_config, k_value, query_files=None):
    """Evaluates embeddings for a single combination of parameters."""
    start_time = time.time()

    agg = eval_params["agg"]
    threshold = eval_params.get("threshold")
    exclude_self_matches = benchmark_config.get("exclude_self_matches", False)

    # Filter ground truth to only include queries that are in the query_files list
    if query_files:
        filtered_ground_truth = {}
        for query, relevants in ground_truth.items():
            query_basename = os.path.basename(query)
            if query_basename in query_files:
                filtered_ground_truth[query] = relevants
        
        print(f"Filtered ground truth from {len(ground_truth)} to {len(filtered_ground_truth)} queries based on query files.")
        ground_truth = filtered_ground_truth

    eval_kwargs = {"agg": agg, "k": k_value, "exclude_self_matches": exclude_self_matches}
    if agg == "None":
        if threshold is not None:
            eval_kwargs["threshold"] = threshold
        else:
            print(f"Warning: agg='None' requires 'threshold' in eval_params. Provided: {eval_params}. Skipping this combo.", file=sys.stderr)
            return None, 0

    print(f"Evaluating with k={k_value}, Agg={agg}" + (f", Thr={threshold}" if agg == "None" else ""))

    try:
        evaluation_output = evaluate_benchmark_embeddings(
            datalake_embeddings=datalake_embeddings,
            ground_truth=ground_truth,
            query_embeddings=query_embeddings,
            **eval_kwargs
        )

        if not isinstance(evaluation_output, dict) or \
           'metrics' not in evaluation_output or \
           'detailed_results' not in evaluation_output:
            print("Error: evaluate_benchmark_embeddings did not return the expected dictionary structure.", file=sys.stderr)
            raise TypeError("Invalid return structure from evaluation function")

    except Exception as e:
        print(f"Error during evaluation with params {eval_params}: {e}", file=sys.stderr)
        raise

    evaluation_time = time.time() - start_time
    print(f"Evaluation finished in {evaluation_time:.2f} seconds.")

    return evaluation_output, evaluation_time

def process_combination(benchmark_name, embedding_type, embed_params, eval_params, 
                        benchmark_config, k_value, query_files):
    """
    Processes a single combination of embedding and evaluation parameters.
    Saves results to JSON.
    """
    result_data = {
        "benchmark": benchmark_name,
        "embedding_type": embedding_type,
        "embedding_params": embed_params,
        "evaluation_params": eval_params,
        "k_value": k_value,
        "status": "started",
        "error": None,
        "evaluation_results": None,  # Will hold dict from evaluate_single_combination
        "times": {},
        "file_paths": {}
    }

    output_dir = f'results/{benchmark_name}'
    ensure_dir(output_dir)
    filename = generate_filename(benchmark_name, embedding_type, embed_params, eval_params)
    output_path = os.path.join(output_dir, filename)

    try:
        # 1. Generate/Load Embeddings
        datalake_embeddings, query_embeddings, gen_time, dl_path, q_path = generate_embeddings(
            benchmark_name, embedding_type, embed_params, query_files
        )
        result_data["times"]["offline"] = gen_time
        result_data["file_paths"]["datalake_embeddings"] = dl_path
        result_data["file_paths"]["query_embeddings"] = q_path
        result_data["status"] = "embeddings_loaded" if gen_time == 0 else "embeddings_generated"

        # 2. Load Ground Truth
        ground_truth = load_ground_truth(benchmark_name)

        # 3. Evaluate
        eval_output, eval_time = evaluate_single_combination(
            datalake_embeddings, query_embeddings, ground_truth,
            eval_params, benchmark_config, k_value, query_files
        )

        # Check if evaluation was skipped (e.g., missing threshold for agg='None')
        if eval_output is None:
            result_data["status"] = "evaluation_skipped"
            result_data["error"] = "Evaluation prerequisites not met (e.g., missing threshold)."
        else:
            result_data["times"]["online"] = eval_time
            result_data["evaluation_results"] = eval_output  # Store the entire result dict
            result_data["status"] = "completed"

    except Exception as e:
        import traceback  # Include traceback for better debugging
        print(f"Failed combination: {embedding_type}, {embed_params}, {eval_params}. Error: {e}", file=sys.stderr)
        result_data["status"] = "failed"
        result_data["error"] = str(e)
        result_data["traceback"] = traceback.format_exc()  # Add traceback

    # 4. Save Result JSON
    try:
        with open(output_path, 'w') as f:
            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, (pd.Timestamp, pd.Period)): return str(obj)
                try: return str(obj)
                except Exception: return f"Unserializable Type: {type(obj)}"

            json.dump(result_data, f, indent=2, default=default_serializer)
        print(f"Result saved to: {output_path}")
    except Exception as e_save:
        print(f"Error saving result JSON to {output_path}: {e_save}", file=sys.stderr)

def run_benchmark(benchmark_name, embedding_types, k_value=10, limit_combinations=None):
    """Runs the benchmark evaluation for specified methods and parameters."""
    print(f"\n{'='*80}\nStarting benchmark: {benchmark_name}\n{'='*80}")

    benchmark_config = BENCHMARK_CONFIGS.get(benchmark_name, {"exclude_self_matches": False})
    query_files = get_query_files(benchmark_name)
    if not query_files:
        print(f"No query files found for benchmark {benchmark_name}. Exiting.", file=sys.stderr)
        return

    all_combinations = []
    total_combinations = 0

    # --- Prepare all combinations ---
    for embedding_type in embedding_types:
        if embedding_type not in METHOD_GRIDS:
            print(f"Warning: No configuration found for method '{embedding_type}'. Skipping.")
            continue

        print(f"\n--- Preparing combinations for: {embedding_type} ---")
        method_grid = METHOD_GRIDS[embedding_type]
        embed_grid = method_grid.get("embedding_params", {})
        eval_grid = method_grid.get("eval_params", {})

        if embedding_type == "sbert":
            base_params = {k: v for k, v in embed_grid.items() if k not in ["include_names", "names_only"]}
            base_combinations = list(parameter_combinations(base_params))
            valid_name_combinations = [
                {"include_names": True, "names_only": False},
                # {"include_names": True, "names_only": True},
                # {"include_names": False, "names_only": False}
            ]
            embed_combinations = []
            for base_combo in base_combinations:
                for name_combo in valid_name_combinations:
                    embed_combinations.append({**base_combo, **name_combo})
        else:
            embed_combinations = list(parameter_combinations(embed_grid))

        if limit_combinations and len(embed_combinations) > limit_combinations:
            print(f"Limiting {embedding_type} to {limit_combinations} random embedding parameter combinations.")
            indices = np.random.choice(len(embed_combinations), limit_combinations, replace=False)
            embed_combinations = [embed_combinations[i] for i in indices]

        for embed_params in embed_combinations:
            current_eval_combinations = []
            aggs = eval_grid.get("agg", [])
            thresholds = eval_grid.get("threshold", [])

            if embedding_type == "sbert" and embed_params.get("orientation") == "horizontal":
                # Filter out 'None' agg for horizontal SBERT
                allowed_aggs = [a for a in aggs if a != "None"]
                if not allowed_aggs:
                    print(f"Warning: No non-'None' aggregation defined for SBERT horizontal. Skipping {embed_params}")
                    continue
                for agg in allowed_aggs:
                    current_eval_combinations.append({"agg": agg})  # Thresholds irrelevant here
            else:
                # Normal handling for others and SBERT vertical
                for agg in aggs:
                    if agg == "None":
                        if not thresholds:
                            print(f"Warning: agg='None' requested for {embedding_type} but no thresholds defined. Skipping bipartite for {embed_params}.")
                            continue
                        for threshold in thresholds:
                            current_eval_combinations.append({"agg": agg, "threshold": threshold})
                    else:
                        current_eval_combinations.append({"agg": agg})  # Thresholds irrelevant here

            if not current_eval_combinations:
                print(f"Warning: No valid evaluation parameters found for {embedding_type} with {embed_params}. Skipping.")
                continue

            for eval_params in current_eval_combinations:
                all_combinations.append((benchmark_name, embedding_type, embed_params, eval_params,
                                        benchmark_config, k_value, query_files))
                total_combinations += 1

    print(f"\nTotal parameter combinations to process: {total_combinations}")
    if total_combinations == 0:
        print("No valid combinations found based on configurations. Exiting.")
        return

    # --- Execute Combinations Sequentially ---
    print(f"\n--- Running {total_combinations} combinations sequentially ---")
    for i, args_tuple in enumerate(all_combinations):
        print(f"\n--- Processing Combination {i+1}/{total_combinations} ---")
        try:
            process_combination(*args_tuple)
        except Exception as e:  # Catch unexpected errors during the call itself
            print(f"Critical error processing combination {args_tuple}: {e}", file=sys.stderr)

    # --- Final Status Update ---
    print(f"\n{'='*80}")
    print(f"Benchmark Run Completed: {benchmark_name}")
    print(f"Total combinations processed: {total_combinations}")
    print(f"Check logs and 'results/{benchmark_name}/' directory for individual JSON outputs.")
    print(f"{'='*80}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Run table embedding benchmarks, saving detailed output per combination.')
    parser.add_argument('benchmark', type=str, help='Benchmark name (e.g., santos, tus)')
    parser.add_argument('--methods', nargs='+', default=['hash', 'count', 'tfidf', 'sbert'],
                        help='List of embedding methods to evaluate')
    parser.add_argument('--k', type=int, default=10, help='k value for evaluation metrics (P@k, R@k, etc.)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of embedding parameter combinations per method (randomly selected, no seed)')

    args = parser.parse_args()

    if args.benchmark not in BENCHMARK_CONFIGS:
        print(f"Warning: Benchmark '{args.benchmark}' not in BENCHMARK_CONFIGS. Ensure data exists. Using default config if needed.")

    valid_methods = []
    for method in args.methods:
        if method not in METHOD_GRIDS:
            print(f"Warning: Method '{method}' not found in METHOD_GRIDS. Skipping.")
        else:
            valid_methods.append(method)

    if not valid_methods:
        print("Error: No valid methods specified or found in METHOD_GRIDS. Exiting.", file=sys.stderr)
        sys.exit(1)

    run_benchmark(
        benchmark_name=args.benchmark,
        embedding_types=valid_methods,
        k_value=args.k,
        limit_combinations=args.limit
    )

if __name__ == "__main__":
    main()