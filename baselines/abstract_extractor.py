# baselines/abstract_extractor.py

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from tqdm import tqdm
from abc import ABC, abstractmethod
import csv

# Increase CSV field size limit to handle large files
csv.field_size_limit(100000000)

class AbstractEmbeddingExtractor(ABC):
    """
    Abstract base class for all embedding extraction methods.
    
    This class defines the common interface and implements shared functionality
    for extracting embeddings from tabular data.
    """
    
    def __init__(self, sample_size: int = 5, include_column_names: bool = True):
        """
        Initialize the embedding extractor.
        
        Args:
            sample_size: Number of values to sample from each column
            include_column_names: Whether to include column names in embeddings
        """
        self.sample_size = sample_size
        self.include_column_names = include_column_names
        
    def collect_column_texts(self, input_dir: str, selected_files: Optional[List[str]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
        """
        Collect text representations of columns from CSV files in the directory.
        
        Args:
            input_dir: Directory containing CSV files
            selected_files: Optional list of specific files to process
            
        Returns:
            Tuple containing:
            - List of table names
            - List of lists of column texts (one list per table)
            - Flattened list of all column texts
        """
        all_column_texts = []
        table_names = []
        table_column_texts = []
        
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        
        if selected_files:
            csv_files = [f for f in csv_files if f in selected_files]
        
        for csv_file in tqdm(csv_files, desc="Collecting column texts"):
            file_path = os.path.join(input_dir, csv_file)
            try:
                df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                
                column_texts = []
                for col in df.columns:
                    non_null_values = df[col].dropna().astype(str)
                    
                    if non_null_values.empty:
                        text = ""
                    else:
                        n = min(self.sample_size, len(non_null_values))
                        samples = non_null_values.sample(n=n, random_state=42).tolist()
                        text = " ".join(samples)
                    
                    # Add column name if specified
                    if self.include_column_names:
                        text = col + " " + text if text else col
                    
                    column_texts.append(text)
                
                table_names.append(table_name)
                table_column_texts.append(column_texts)
                all_column_texts.extend(column_texts)
                
            except Exception as e:
                print(f"Error collecting texts from {file_path}: {e}")
        
        return table_names, table_column_texts, all_column_texts
    
    @abstractmethod
    def build_vectorizer(self, texts: List[str]) -> Any:
        """
        Build and return a vectorizer trained on the provided texts.
        
        Args:
            texts: List of text strings to train the vectorizer on
            
        Returns:
            Trained vectorizer object
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str], vectorizer: Any) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the vectorizer.
        
        Args:
            texts: List of text strings to generate embeddings for
            vectorizer: Trained vectorizer to use
            
        Returns:
            NumPy array of embeddings
        """
        pass
    
    def process_directory(
        self,
        input_dir: str,
        output_path: str,
        additional_vocab_dirs: Optional[List[str]] = None,
        selected_files: Optional[List[str]] = None,
        vectorizer: Optional[Any] = None
    ) -> Any:
        """
        Process a directory of CSV files to extract embeddings.
        
        Args:
            input_dir: Directory containing CSV files
            output_path: Path to save embeddings pickle file
            additional_vocab_dirs: Additional directories for vocabulary building
            selected_files: Optional list of specific files to process
            vectorizer: Pre-built vectorizer (optional)
            
        Returns:
            The trained vectorizer
        """
        # Step 1: Collect texts
        table_names, table_column_texts, all_column_texts = self.collect_column_texts(
            input_dir, selected_files
        )
        
        # Step 2: Build or reuse vectorizer
        if vectorizer is None:
            print(f"Building shared vocabulary...")
            
            # If additional directories are specified, collect their texts too
            if additional_vocab_dirs:
                additional_texts = []
                vocab_dirs = additional_vocab_dirs
                if isinstance(vocab_dirs, str):
                    vocab_dirs = [vocab_dirs]
                    
                for directory in vocab_dirs:
                    _, _, directory_texts = self.collect_column_texts(directory)
                    additional_texts.extend(directory_texts)
                    
                all_column_texts.extend(additional_texts)
            
            # Build vectorizer
            vectorizer = self.build_vectorizer(all_column_texts)
        
        # Step 3: Generate embeddings
        name_mode = "with" if self.include_column_names else "without"
        print(f"Generating embeddings for {input_dir} {name_mode} column names")
        
        table_embeddings = []
        for table_name, column_texts in zip(table_names, table_column_texts):
            try:
                embeddings = self.generate_embeddings(column_texts, vectorizer)
                table_embeddings.append((table_name, embeddings))
            except Exception as e:
                print(f"Error processing {table_name}: {e}")
        
        # Step 4: Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(table_embeddings, f)
        
        print(f"Saved embeddings to {output_path}")
        return vectorizer