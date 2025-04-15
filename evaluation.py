"""
This script evaluates how well table embeddings can match related tables in a benchmark dataset. 
It works with two main approaches: you can either combine all column embeddings into a single 
table vector (using mean, max, or sum aggregation) or compare columns directly using bipartite 
matching. The aggregation approach is faster and works well for simple cases, while bipartite 
matching captures more nuanced column relationships but is computationally more intensive.

When using aggregation, the script leverages FAISS for efficient similarity search, making it 
scalable to large candidate sets. With bipartite matching (when agg='None'), it uses the Hungarian 
algorithm to find the optimal alignment between columns across tables, applying a similarity 
threshold to focus on meaningful matches.

The evaluation calculates standard IR metrics (precision, recall, F1, and MAP) at different 
k values, comparing the retrieved candidates against ground truth relevance judgments. The 
results include both aggregate metrics and detailed per-query data, letting you analyze 
performance across different query types or difficulty levels.
"""

import numpy as np
import faiss
from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm as numpy_norm
from tqdm import tqdm

def cosine_sim(vec1, vec2):
    norm1 = numpy_norm(vec1)
    norm2 = numpy_norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def evaluate_benchmark_embeddings(datalake_embeddings, ground_truth, query_embeddings=None,
                                  agg='mean', k=10, sample_size=None, threshold=0.1,
                                  exclude_self_matches=True):
    """
    Evaluate benchmark embeddings using aggregated table embeddings or bipartite matching,
    and a ground truth relevance mapping, using FAISS for fast similarity search
    with cosine similarity when applicable.

    Parameters:
        datalake_embeddings (list of tuples): Each tuple (table_name, col_embeddings).
        ground_truth (dict): Query table names map to lists of relevant table names.
        query_embeddings (list of tuples, optional): Embeddings for query tables.
        agg (str): 'mean', 'max', 'sum', or 'None' (for bipartite matching).
        k (int): Number of top results to consider.
        sample_size (int, optional): Number of queries to randomly sample.
        threshold (float): Min similarity for bipartite matching edges.
        exclude_self_matches (bool): Exclude query table from its own results.

    Returns:
        dict: A dictionary containing:
              - 'metrics': Average P@k, R@k, F1@k, MAP@k.
              - 'detailed_results': Per-query candidates and ground truth.
                {query_name: {'candidates': [...], 'ground_truth': [...]}, ...}
    """
    # Process the ground truth to remove '.csv'
    gt_processed = {}
    for query, rel_tables in ground_truth.items():
        query_clean = query.replace('.csv', '')
        rel_clean = [t.replace('.csv', '') for t in rel_tables]
        gt_processed[query_clean] = rel_clean

    def compute_bm_score(table1_emb, table2_emb, threshold=threshold):
        score = 0.0
        if table1_emb is None or table2_emb is None or table1_emb.shape[0] == 0 or table2_emb.shape[0] == 0:
             return 0.0
        nrow = table1_emb.shape[0]
        ncol = table2_emb.shape[0]
        graph = np.zeros(shape=(nrow, ncol), dtype=float)

        for i in range(nrow):
            for j in range(ncol):
                sim = cosine_sim(table1_emb[i], table2_emb[j])
                if sim > threshold:
                    graph[i, j] = sim

        if np.all(graph <= threshold):
             return 0.0

        try:
            cost_matrix = graph.max() - graph
            cost_matrix[graph <= threshold] = DISALLOWED

            m = Munkres()
            if np.any(np.isfinite(cost_matrix)):
                indexes = m.compute(cost_matrix)
                for row, col in indexes:
                    if graph[row, col] > threshold:
                        score += graph[row, col]
        except Exception as e:
            print(f"Warning: Munkres computation failed: {e}. Returning score 0.0 for this pair.")
            return 0.0

        return score

    # Prepare embeddings based on aggregation method
    table_embeds = {}
    col_embeds = {}
    query_to_embedding = {}
    query_to_col_embedding = {}
    datalake_table_names_for_bm = []

    if agg == 'None':
        print("Preparing embeddings for Bipartite Matching...")
        all_embeddings_source = list(datalake_embeddings)
        if query_embeddings is not None:
            dl_names = {name.replace('.csv','') for name, _ in datalake_embeddings}
            for q_name, q_emb in query_embeddings:
                q_name_clean = q_name.replace('.csv','')
                if q_name_clean not in dl_names:
                     all_embeddings_source.append((q_name, q_emb))

        for table_name, embeddings_array in all_embeddings_source:
            table_clean = table_name.replace('.csv', '')
            if embeddings_array is not None and embeddings_array.size > 0:
                col_embeds[table_clean] = embeddings_array
                datalake_table_names_for_bm.append(table_clean)
            else:
                 print(f"Warning: Skipping table '{table_clean}' due to missing or empty embeddings.")

        query_col_embeds_provided = {}
        if query_embeddings is not None:
             for q_name, q_emb in query_embeddings:
                  q_name_clean = q_name.replace('.csv', '')
                  if q_emb is not None and q_emb.size > 0:
                      query_col_embeds_provided[q_name_clean] = q_emb
                  else:
                      print(f"Warning: Skipping query '{q_name_clean}' due to missing or empty provided embeddings.")

        for query_clean in gt_processed:
             if query_clean in query_col_embeds_provided:
                 query_to_col_embedding[query_clean] = query_col_embeds_provided[query_clean]
             elif query_clean in col_embeds:
                 query_to_col_embedding[query_clean] = col_embeds[query_clean]

    else:
        print(f"Aggregating embeddings using method: {agg}...")
        for table_name, embeddings_array in datalake_embeddings:
            table_clean = table_name.replace('.csv', '')
            if embeddings_array is None or embeddings_array.size == 0:
                 print(f"Warning: Skipping datalake table '{table_clean}' due to missing or empty embeddings.")
                 continue
            try:
                if agg == 'mean': agg_emb = np.mean(embeddings_array, axis=0)
                elif agg == 'max': agg_emb = np.max(embeddings_array, axis=0)
                elif agg == 'sum': agg_emb = np.sum(embeddings_array, axis=0)
                else: raise ValueError(f"Unsupported aggregation method: {agg}")
                table_embeds[table_clean] = agg_emb
            except Exception as e:
                 print(f"Error aggregating embeddings for table {table_clean}: {e}")

        query_table_embeds_provided = {}
        if query_embeddings is not None:
            for table_name, embeddings_array in query_embeddings:
                table_clean = table_name.replace('.csv', '')
                if embeddings_array is None or embeddings_array.size == 0:
                     print(f"Warning: Skipping query table '{table_clean}' due to missing or empty provided embeddings.")
                     continue
                try:
                    if agg == 'mean': agg_emb = np.mean(embeddings_array, axis=0)
                    elif agg == 'max': agg_emb = np.max(embeddings_array, axis=0)
                    elif agg == 'sum': agg_emb = np.sum(embeddings_array, axis=0)
                    else: raise ValueError(f"Unsupported aggregation method: {agg}")
                    query_table_embeds_provided[table_clean] = agg_emb
                except Exception as e:
                    print(f"Error aggregating embeddings for query {table_clean}: {e}")

        for query_clean in gt_processed:
             if query_clean in query_table_embeds_provided:
                 query_to_embedding[query_clean] = query_table_embeds_provided[query_clean]
             elif query_clean in table_embeds:
                 query_to_embedding[query_clean] = table_embeds[query_clean]

        all_table_embeds = table_embeds.copy()
        all_table_embeds.update(query_table_embeds_provided)

        table_names = list(all_table_embeds.keys())
        if not table_names:
             print("Error: No valid aggregated embeddings found to build FAISS index.")
             return {'metrics': {}, 'detailed_results': {}}

        table_vectors = np.stack([all_table_embeds[name] for name in table_names]).astype('float32')

        # Normalize for cosine similarity search via inner product
        faiss.normalize_L2(table_vectors)
        name_to_idx = {name: idx for idx, name in enumerate(table_names)}

        # Build FAISS index
        d = table_vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(table_vectors)

    # Filter valid queries
    if agg == 'None':
        valid_queries = [q for q in gt_processed if q in query_to_col_embedding]
    else:
        valid_queries = [q for q in gt_processed if q in query_to_embedding]

    if not valid_queries:
        print("Error: No valid queries found with embeddings matching ground truth.")
        return {'metrics': {}, 'detailed_results': {}}

    if sample_size is not None and sample_size < len(valid_queries):
        print(f"Sampling {sample_size} queries from {len(valid_queries)} valid queries.")
        np.random.seed(42)
        indices = np.random.choice(len(valid_queries), size=sample_size, replace=False)
        valid_queries = [valid_queries[i] for i in indices]

    print(f"Evaluating on {len(valid_queries)} queries.")

    # Evaluation metrics initialization
    p_scores = {i: [] for i in range(1, k + 1)}
    r_scores = {i: [] for i in range(1, k + 1)}
    f1_scores = {i: [] for i in range(1, k + 1)}
    ap_scores = {i: [] for i in range(1, k + 1)}
    detailed_query_results = {}

    for query in tqdm(valid_queries, desc="Processing queries"):
        relevant = set(gt_processed[query])
        ranked_names = []
        scores_dict = {}

        if agg == 'None':
            query_col_embs = query_to_col_embedding.get(query)
            if query_col_embs is None: continue

            scores = []
            candidate_tables = list(col_embeds.keys())
            for table_clean in candidate_tables:
                if exclude_self_matches and table_clean == query:
                    continue
                table_col_embs = col_embeds.get(table_clean)
                if table_col_embs is None: continue

                bm_score = compute_bm_score(query_col_embs, table_col_embs, threshold)
                scores.append((bm_score, table_clean))

            scores.sort(reverse=True)
            ranked_names = [name for score, name in scores[:k]]
            scores_dict = {name: score for score, name in scores[:k]}

        else:
            if query not in name_to_idx: continue
            q_idx = name_to_idx[query]
            q_vector = table_vectors[q_idx].reshape(1, -1)

            additional = 1 if exclude_self_matches else 0
            try:
                D, I = index.search(q_vector, k + additional)
            except Exception as e:
                 print(f"FAISS search failed for query {query}: {e}")
                 continue

            retrieved_indices = I[0]
            retrieved_scores = D[0]
            processed_results = []

            for idx, score in zip(retrieved_indices, retrieved_scores):
                 if idx == -1: continue
                 table_name = table_names[idx]
                 if exclude_self_matches and table_name == query:
                     continue
                 processed_results.append((table_name, score))
                 if len(processed_results) == k:
                     break

            ranked_names = [name for name, score in processed_results]
            scores_dict = {name: score for name, score in processed_results}

        detailed_query_results[query] = {
            "candidates": ranked_names,
            "ground_truth": list(relevant)
        }

        # Compute metrics (P, R, F1, AP) at each cutoff
        num_relevant_retrieved = 0
        sum_precision_at_k = 0.0
        for i in range(1, k + 1):
            if i > len(ranked_names):
                 p = num_relevant_retrieved / len(ranked_names) if len(ranked_names) > 0 else 0.0
                 r = num_relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
                 ap_value = sum_precision_at_k / min(i, len(relevant)) if min(i, len(relevant)) > 0 else 0.0
            else:
                candidate_at_i = ranked_names[i-1]
                is_relevant = candidate_at_i in relevant
                if is_relevant:
                    num_relevant_retrieved += 1
                    sum_precision_at_k += (num_relevant_retrieved / i)

                p = num_relevant_retrieved / i
                r = num_relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
                ap_value = sum_precision_at_k / min(i, len(relevant)) if min(i, len(relevant)) > 0 else 0.0

            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            p_scores[i].append(p)
            r_scores[i].append(r)
            f1_scores[i].append(f1)
            ap_scores[i].append(ap_value)

    # Aggregate and format results
    results_metrics = {}
    print("\nEvaluation Results")
    print("------------------")
    print(f"Aggregation Method   : {agg}")
    print(f"Top-k                : {k}")
    if agg == 'None':
        print(f"Bipartite Match Threshold: {threshold}")
    print(f"Exclude Self-Matches : {exclude_self_matches}")
    print(f"Number of Queries    : {len(valid_queries)}\n")
    print("{:<4} {:<12} {:<12} {:<12} {:<12}".format("k", "Precision", "Recall", "F1", "MAP"))
    print("-" * 60)

    for i in range(1, k + 1):
        p_avg = np.mean(p_scores[i]) if p_scores[i] else 0.0
        r_avg = np.mean(r_scores[i]) if r_scores[i] else 0.0
        f1_avg = np.mean(f1_scores[i]) if f1_scores[i] else 0.0
        map_avg = np.mean(ap_scores[i]) if ap_scores[i] else 0.0

        results_metrics[f'p@{i}'] = p_avg
        results_metrics[f'r@{i}'] = r_avg
        results_metrics[f'f1@{i}'] = f1_avg
        results_metrics[f'map@{i}'] = map_avg
        print("{:<4} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(i, p_avg, r_avg, f1_avg, map_avg))
    print("-" * 60)

    final_output = {
        "metrics": results_metrics,
        "detailed_results": detailed_query_results
    }
    return final_output