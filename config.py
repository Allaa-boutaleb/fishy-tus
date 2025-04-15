# config.py

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
            "deduplicate": [True],
            "include_names": [True, False],
            "names_only": [True, False]
        },
        "eval_params": {
            "agg": ["mean"],
            "threshold": [0.1]  # Only used when agg="None"
        }
    }
}