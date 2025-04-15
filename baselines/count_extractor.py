# baselines/count_extractor.py

import os
import numpy as np
from typing import List, Optional, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from baselines.abstract_extractor import AbstractEmbeddingExtractor

class CountEmbeddingExtractor(AbstractEmbeddingExtractor):
    """
    Extracts count-based embeddings from tabular data.
    """
    
    def __init__(
        self, 
        sample_size: int = 5, 
        max_features: int = 1024, 
        ngram_range: tuple = (1, 1),
        include_column_names: bool = True
    ):
        """
        Initialize the count-based embedding extractor.
        
        Args:
            sample_size: Number of values to sample from each column
            max_features: Maximum size of vocabulary for CountVectorizer
            ngram_range: Range of n-grams to consider
            include_column_names: Whether to include column names in text representation
        """
        super().__init__(sample_size, include_column_names)
        self.max_features = max_features
        self.ngram_range = ngram_range
        
    def build_vectorizer(self, texts: List[str]) -> CountVectorizer:
        """
        Build and return a CountVectorizer trained on the provided texts.
        
        Args:
            texts: List of text strings to train the vectorizer on
            
        Returns:
            Trained CountVectorizer
        """
        print(f"Fitting CountVectorizer (max_features={self.max_features})")
        vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b',
            binary=False,
            ngram_range=self.ngram_range
        )
        vectorizer.fit(texts)
        return vectorizer
    
    def generate_embeddings(self, texts: List[str], vectorizer: CountVectorizer) -> np.ndarray:
        """
        Generate count-based embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            vectorizer: Trained CountVectorizer to use
            
        Returns:
            NumPy array of normalized count embeddings
        """
        X = vectorizer.transform(texts).toarray()
        return normalize(X, norm='l2', axis=1)
    
    def process_directory(
        self,
        input_dir: str,
        output_path: str,
        additional_vocab_dirs: Optional[List[str]] = None,
        selected_files: Optional[List[str]] = None,
        vectorizer: Optional[CountVectorizer] = None
    ) -> CountVectorizer:
        """
        Process a directory of CSV files to extract count-based embeddings.
        
        Args:
            input_dir: Directory containing CSV files
            output_path: Path to save embeddings pickle file
            additional_vocab_dirs: Additional directories for vocabulary building
            selected_files: Optional list of specific files to process
            vectorizer: Pre-built CountVectorizer (optional)
            
        Returns:
            The trained CountVectorizer
        """
        return super().process_directory(input_dir, output_path, additional_vocab_dirs, selected_files, vectorizer)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate count-based embeddings for CSV columns')
    parser.add_argument('input_dir', help='Directory containing input CSV files')
    parser.add_argument('output_path', help='Path to save output embeddings')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples to take from each column')
    parser.add_argument('--max_features', type=int, default=1024, help='Maximum vocabulary size')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1, help='Maximum n-gram size')
    parser.add_argument('--additional_vocab_dirs', nargs='+', help='Additional directories for vocabulary building')
    parser.add_argument('--include_column_names', action='store_true', default=True, 
                      help='Include column names in text representations (default: True)')
    parser.add_argument('--exclude_column_names', dest='include_column_names', action='store_false',
                      help='Exclude column names from text representations')
    
    args = parser.parse_args()
    
    extractor = CountEmbeddingExtractor(
        sample_size=args.sample_size,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        include_column_names=args.include_column_names
    )
    
    extractor.process_directory(
        input_dir=args.input_dir,
        output_path=args.output_path,
        additional_vocab_dirs=args.additional_vocab_dirs,
        selected_files=None
    )