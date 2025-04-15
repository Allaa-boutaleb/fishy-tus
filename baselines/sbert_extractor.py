# baselines/sbert_extractor.py

import os
import numpy as np
from typing import List, Optional, Any, Tuple, Dict
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from tqdm import tqdm

from baselines.abstract_extractor import AbstractEmbeddingExtractor

class SbertEmbeddingExtractor(AbstractEmbeddingExtractor):
    """
    Extracts SBERT embeddings from tabular data.
    
    This class is a special case that doesn't fully inherit from AbstractEmbeddingExtractor
    since the SBERT embedding process is significantly different.
    """
    
    def __init__(
        self, 
        sample_size: int = 10, 
        model_name: str = "all-mpnet-base-v2",
        orientation: str = "vertical",
        include_names: bool = True,
        names_only: bool = False,
        deduplicate: bool = True,
        batch_size: int = 256
    ):
        """
        Initialize the SBERT embedding extractor.
        
        Args:
            sample_size: Number of values to sample per column/row
            model_name: Name of the SentenceTransformer model to use
            orientation: "vertical" (column-wise) or "horizontal" (row-wise)
            include_names: Whether to include column names
            names_only: Whether to only use column names (only applies when include_names=True)
            deduplicate: Whether to deduplicate values before sampling
            batch_size: Batch size for sentence transformer encoding
        """
        # Note: we're not calling super().__init__() because SBERT is a special case
        self.sample_size = sample_size
        self.model_name = model_name
        self.orientation = orientation
        
        # Validate parameter combinations
        self.include_names = include_names
        # names_only can only be True if include_names is True
        self.names_only = names_only if include_names else False
        
        self.deduplicate = deduplicate
        self.batch_size = batch_size
        self.model = None
        
    @staticmethod
    def is_valid_parameter_combination(include_names: bool, names_only: bool) -> bool:
        """
        Check if the parameter combination is valid:
        - include_names=True, names_only=True: Valid
        - include_names=True, names_only=False: Valid
        - include_names=False, names_only=True: Invalid
        - include_names=False, names_only=False: Valid
        """
        if not include_names and names_only:
            return False
        return True
        
    def load_model(self):
        """Load the SentenceTransformer model if not already loaded."""
        if self.model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded.")
        return self.model
    
    def build_vectorizer(self, texts: List[str]) -> Any:
        """
        Build and return a vectorizer.
        
        This method exists for API compatibility but isn't used for SBERT.
        """
        return self.load_model()
    
    def generate_embeddings(self, texts: List[str], vectorizer: Any) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        This method exists for API compatibility but isn't used for SBERT.
        """
        model = vectorizer if isinstance(vectorizer, SentenceTransformer) else self.load_model()
        return model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
    
    def generate_table_embeddings(
        self,
        file_path: str
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        Generate embeddings for table columns or rows using SBERT.

        Args:
            file_path: Path to CSV file

        Returns:
            A tuple containing the table name and a NumPy array of embeddings,
            or None if an error occurs or the table is empty/unsuitable.
        """
        model = self.load_model()
        
        try:
            # Use engine='python' and on_bad_lines='skip' for robustness
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
            if df.empty:
                print(f"Skipping empty file: {file_path}")
                return None

            table_name = os.path.splitext(os.path.basename(file_path))[0]

            # Get max sequence length from the model if available, default to 512
            max_seq_length_tok = getattr(model, 'max_seq_length', 512)
            # Heuristic: Use character limit slightly larger than token limit
            max_char_length = max_seq_length_tok * 5

            texts_to_encode = []
            valid_identifiers = []  # To store corresponding column names or row indices

            if self.orientation == "vertical":
                # --- Column-wise embeddings ---
                for col in df.columns:
                    col_text = ""
                    if self.names_only:
                        col_text = f"Column: {col}"
                    else:
                        # Ensure column exists and handle potential multi-index issues safely
                        if col not in df.columns:
                            print(f"Warning: Column '{col}' not found in {file_path}. Skipping.")
                            continue

                        non_null_values = df[col].dropna().astype(str)

                        if non_null_values.empty:
                            if self.include_names:
                                # Represent empty columns with just their name if requested
                                col_text = f"Column: {col}"
                            else:
                                # Skip empty columns if names are not included
                                continue
                        else:
                            sample_values = []
                            if self.deduplicate:
                                unique_values = non_null_values.unique()
                                n_samples = min(self.sample_size, len(unique_values))
                                # Sample directly from unique values
                                sample_values = np.random.choice(unique_values, size=n_samples, replace=False).tolist()
                            else:
                                n_samples = min(self.sample_size, len(non_null_values))
                                # Sample with replacement allowed if sample_size > len(non_null_values)
                                sample_values = non_null_values.sample(
                                    n=n_samples, 
                                    random_state=42, 
                                    replace=len(non_null_values) < n_samples
                                ).tolist()

                            if self.include_names:
                                column_text_data = ", ".join(sample_values)
                                col_text = f"Column: {col}. Values: {column_text_data}"
                            else:
                                col_text = ", ".join(sample_values)

                    # Truncate if needed (apply before adding to list)
                    if len(col_text) > max_char_length:
                        col_text = col_text[:max_char_length]

                    if col_text:  # Ensure we don't add empty strings if skipping empty cols without names
                        texts_to_encode.append(col_text)
                        valid_identifiers.append(col)  # Store the column name

                if not texts_to_encode:
                    print(f"No content to encode for table {table_name} in vertical orientation.")
                    return None

                # Batch Encode All Column Texts
                column_embeddings = model.encode(texts_to_encode, batch_size=self.batch_size, show_progress_bar=False)

                return (table_name, np.array(column_embeddings))  # Embeddings order matches valid_identifiers

            else:  # horizontal (Row-wise serialization)
                if self.names_only:
                    # Special case: Encode only the header
                    if not df.columns.empty:
                        header_text = ", ".join([f"{col}" for col in df.columns])
                        if len(header_text) > max_char_length:
                            header_text = header_text[:max_char_length]
                        texts_to_encode.append(header_text)
                        valid_identifiers.append("header")  # Identifier for the single header embedding
                    # If no columns, texts_to_encode remains empty
                else:
                    n_rows_to_sample = min(self.sample_size, len(df))
                    if n_rows_to_sample == 0:
                        print(f"No rows to sample in {file_path}")
                        return None

                    # Sample row indices first
                    row_indices_to_process = df.sample(n=n_rows_to_sample, random_state=42).index if n_rows_to_sample < len(df) else df.index

                    for idx in row_indices_to_process:
                        row = df.loc[idx]
                        non_null_items = {col: str(val) for col, val in row.items() if pd.notna(val) and str(val).strip()}  # Filter empty strings too

                        if not non_null_items:
                            continue  # Skip rows with no non-null values

                        row_text = ""
                        if self.include_names:
                            row_text = ", ".join([f"{col}: {val}" for col, val in non_null_items.items()])
                        else:
                            row_text = ", ".join(non_null_items.values())

                        # Truncate if needed
                        if len(row_text) > max_char_length:
                            row_text = row_text[:max_char_length]

                        texts_to_encode.append(row_text)
                        valid_identifiers.append(idx)  # Store the original row index

                if not texts_to_encode:
                    print(f"No content to encode for table {table_name} in horizontal orientation.")
                    return None

                # Batch Encode All Row Texts 
                row_embeddings = model.encode(texts_to_encode, batch_size=self.batch_size, show_progress_bar=False)

                return (table_name, np.array(row_embeddings))

        except pd.errors.EmptyDataError:
            print(f"Skipping empty or invalid CSV file: {file_path}")
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Consider more specific error handling if needed
            import traceback
            traceback.print_exc()
            return None
    
    def process_directory(
        self,
        input_dir: str,
        output_path: str,
        additional_vocab_dirs: Optional[List[str]] = None,
        selected_files: Optional[List[str]] = None,
        vectorizer: Optional[Any] = None
    ) -> Any:
        """
        Process directory of CSV files, generate embeddings using batch SBERT, and save.
        
        Args:
            input_dir: Directory containing CSV files
            output_path: Path to save embeddings pickle file
            additional_vocab_dirs: Additional directories (unused)
            selected_files: Optional list of specific files to process
            vectorizer: Pre-built vectorizer (unused)
            
        Returns:
            The SentenceTransformer model
        """
        # We don't call super().process_directory() because SBERT is a special case
        
        model = self.load_model()
        
        table_embeddings_list = []
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

        if selected_files:
            # Filter based on the provided list, ensuring '.csv' if needed
            selected_files_set = set(sf if sf.endswith('.csv') else sf + '.csv' for sf in selected_files)
            csv_files_to_process = [f for f in all_files if f in selected_files_set]
            if not csv_files_to_process:
                print(f"Warning: No specified files found in {input_dir}.")
                return model
            print(f"Processing {len(csv_files_to_process)} selected tables.")
        else:
            csv_files_to_process = all_files
            print(f"Processing all {len(csv_files_to_process)} CSV tables found in {input_dir}.")

        print(f"Configuration:")
        print(f"  Orientation: {self.orientation}")
        print(f"  Include names: {self.include_names}")
        print(f"  Names only: {self.names_only}")
        print(f"  Sample size: {self.sample_size}")
        print(f"  Deduplicate (vertical only): {self.deduplicate}")
        print(f"  Batch size: {self.batch_size}")

        # Use tqdm for progress bar over the file iteration
        for csv_file in tqdm(csv_files_to_process, desc="Processing tables"):
            file_path = os.path.join(input_dir, csv_file)
            result = self.generate_table_embeddings(file_path)
            if result:
                table_embeddings_list.append(result)

        if not table_embeddings_list:
            print("No embeddings were generated.")
            return model

        # --- Save embeddings and metadata ---
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Ensure directory exists only if output_path includes a directory
            os.makedirs(output_dir, exist_ok=True)

        metadata = {
            "model_name": self.model_name,
            "orientation": self.orientation,
            "include_names": self.include_names,
            "names_only": self.names_only,
            "sample_size": self.sample_size,
            "deduplicate": self.deduplicate,
            "batch_size": self.batch_size
        }

        try:
            with open(output_path, 'wb') as f:
                pickle.dump((table_embeddings_list, metadata), f)
            print(f"\nSaved embeddings for {len(table_embeddings_list)} tables to {output_path}")
        except Exception as e:
            print(f"\nError saving embeddings to {output_path}: {e}")
            
        return model


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBERT embeddings for CSV tables using batch processing.")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("output_path", help="Output pickle file path for embeddings and metadata")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of sample values/rows per column/table (default: 10)")
    parser.add_argument("--model_name", default="all-mpnet-base-v2", help="SentenceTransformer model name (default: all-mpnet-base-v2)")
    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical", help="Embedding orientation: 'vertical' (column-wise) or 'horizontal' (row-wise) (default: vertical)")
    parser.add_argument("--include_names", action=argparse.BooleanOptionalAction, default=True, help="Include column names in the text representation (default: True)")
    parser.add_argument("--names_only", action="store_true", help="Only use column/header names for embeddings (ignores values/sample_size)")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True, help="Deduplicate values before sampling (vertical orientation only) (default: True)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for SentenceTransformer encoding (default: 256)")
    parser.add_argument("--selected_files", nargs='*', help="Optional list of specific CSV filenames (without path) to process.")

    args = parser.parse_args()

    # Input validation
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        exit(1)
    if args.names_only and args.orientation == "horizontal":
        print("Info: Using --names_only with --orientation horizontal will embed only the table header.")
    if args.names_only and not args.include_names:
        print("Warning: --names_only requires --include_names. Setting --include_names=True.")
        args.include_names = True  # Force include_names if names_only is set

    extractor = SbertEmbeddingExtractor(
        sample_size=args.sample_size,
        model_name=args.model_name,
        orientation=args.orientation,
        include_names=args.include_names,
        names_only=args.names_only,
        deduplicate=args.deduplicate,
        batch_size=args.batch_size
    )
    
    extractor.process_directory(
        input_dir=args.input_dir,
        output_path=args.output_path,
        selected_files=args.selected_files
    )