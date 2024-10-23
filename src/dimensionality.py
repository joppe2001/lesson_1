from typing import List, Sequence
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from base_plotter import DimReductionData
from dimvis import ClusterAnalyzer
import click
from joblib import Memory
from pathlib import Path

# Setup cache for expensive computations
memory = Memory(Path.home() / '.cache' / 'whatsapp_analysis', verbose=0)


@memory.cache
def cached_fit_transform_tfidf(texts: Sequence[str]) -> np.ndarray:
    """Cached TF-IDF transformation."""
    vectorizer = TfidfVectorizer(
        min_df=5,  # Increased from 2 to 5 for better performance
        max_features=5000,  # Limit features for better performance
        stop_words='english'
    )
    return vectorizer.fit_transform(texts).toarray()


@memory.cache
def cached_tsne_transform(vectors: np.ndarray) -> np.ndarray:
    """Cached t-SNE transformation."""
    tsne = TSNE(
        n_components=2,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        method='barnes_hut',  # Faster algorithm, O(N log N)
        init='pca',  # Better initialization
        learning_rate='auto',
        perplexity=min(30, len(vectors) - 1)
    )
    return tsne.fit_transform(vectors)

# This should be in your dimensionality.py file:

def process_text_for_viz(texts: List[str],
                        authors: List[str],
                        method: str = 'tsne',
                        random_state: int = 42,
                        sample_size: int = 50000,
                        analyze_clusters: bool = False) -> DimReductionData:  # Added parameter
    """
    Process text data for dimensionality reduction visualization with performance optimizations.

    Args:
        texts: List of text messages
        authors: List of author names
        method: Reduction method ('tsne' or 'pca')
        random_state: Random seed for reproducibility
        sample_size: Maximum number of messages to process (for performance)
        analyze_clusters: Whether to perform cluster analysis (default: False)

    Returns:
        DimReductionData object with embedded points and metadata
    """
    # Sample data if too large
    if len(texts) > sample_size:
        click.echo(f"Sampling {sample_size} messages from {len(texts)} for better performance...")
        indices = np.random.RandomState(random_state).choice(
            len(texts), sample_size, replace=False
        )
        texts = [texts[i] for i in indices]
        authors = [authors[i] for i in indices]

    # Clean texts (basic cleaning for performance)
    clean_texts = [
        text.replace('‎.*omitted', '').split(': ', 1)[-1]
        for text in texts
    ]

    with click.progressbar(
            length=2,
            label='Processing text data'
    ) as bar:
        # Text vectorization
        vectors = cached_fit_transform_tfidf(clean_texts)
        bar.update(1)

        # Dimension reduction
        if method.lower() == 'tsne':
            embedded = cached_tsne_transform(vectors)
            explained_variance = None
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
            embedded = reducer.fit_transform(vectors)
            explained_variance = sum(reducer.explained_variance_ratio_)
        bar.update(1)

    # Add cluster analysis
    if analyze_clusters and len(clean_texts) > 0:
        from dimvis import ClusterAnalyzer  # Import here to avoid circular imports
        click.echo("\nAnalyzing clusters...")
        analyzer = ClusterAnalyzer(clean_texts, embedded)
        analyzer.print_cluster_summary()

    return DimReductionData(
        embedded_data=embedded,
        labels=np.array(authors),
        explained_variance=explained_variance
    )