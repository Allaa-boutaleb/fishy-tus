# baselines/hash_extractor.py

import os
import numpy as np
from typing import List, Optional, Any
from sklearn.feature_extraction.text import HashingVectorizer

from baselines.abstract_extractor import AbstractEmbeddingExtractor

class HashEmbeddingExtractor(AbstractEmbeddingExtractor):
    """
    Extracts hash-based embeddings from tabular data.
    """
    
    def __init__(
        self, 
        sample_size: int = 5, 
        n_features: int = 1024,
        include_column_names: bool = False
    ):
        """
        Initialize the hash-based embedding extractor.
        
        Args:
            sample_size: Number of values to sample from each column
            n_features: Size of hash embedding vector
            include_column_names: Whether to include column names in text representation
        """
        super().__init__(sample_size, include_column_names)
        self.n_features = n_features
        
    def build_vectorizer(self, texts: List[str]) -> HashingVectorizer:
        """
        Build and return a HashingVectorizer.
        
        Note: HashingVectorizer doesn't actually need to be trained on texts,
        but we retain this method for API consistency.
        
        Args:
            texts: List of text strings (unused)
            
        Returns:
            HashingVectorizer instance
        """
        print(f"Creating HashingVectorizer (n_features={self.n_features})")
        return HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm='l2'
        )
    
    def generate_embeddings(self, texts: List[str], vectorizer: HashingVectorizer) -> np.ndarray:
        """
        Generate hash-based embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            vectorizer: HashingVectorizer to use
            
        Returns:
            NumPy array of hash embeddings
        """
        # Filter out empty strings, which would create empty embeddings
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return np.array([])
            
        embeddings = []
        for text in valid_texts:
            embedding = vectorizer.transform([text]).toarray()[0]
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def process_directory(
        self,
        input_dir: str,
        output_path: str,
        additional_vocab_dirs: Optional[List[str]] = None,
        selected_files: Optional[List[str]] = None,
        vectorizer: Optional[HashingVectorizer] = None
    ) -> HashingVectorizer:
        """
        Process a directory of CSV files to extract hash-based embeddings.
        
        Note: additional_vocab_dirs is unused for hash-based embeddings but kept
        for API consistency.
        
        Args:
            input_dir: Directory containing CSV files
            output_path: Path to save embeddings pickle file
            additional_vocab_dirs: Additional directories (unused)
            selected_files: Optional list of specific files to process
            vectorizer: Pre-built HashingVectorizer (optional)
            
        Returns:
            The HashingVectorizer
        """
        return super().process_directory(input_dir, output_path, None, selected_files, vectorizer)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate hash-based embeddings for CSV columns')
    parser.add_argument('input_dir', help='Directory containing input CSV files')
    parser.add_argument('output_path', help='Path to save output embeddings')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples to take from each column')
    parser.add_argument('--n_features', type=int, default=1024, help='Size of hash space')
    parser.add_argument('--include_column_names', action='store_true', 
                       help='Include column names in hash calculation')
    
    args = parser.parse_args()
    
    extractor = HashEmbeddingExtractor(
        sample_size=args.sample_size,
        n_features=args.n_features,
        include_column_names=args.include_column_names
    )
    
    extractor.process_directory(
        input_dir=args.input_dir,
        output_path=args.output_path,
        selected_files=None
    )