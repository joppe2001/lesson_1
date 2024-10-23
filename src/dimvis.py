from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class ClusterAnalyzer:
    def __init__(self, texts: List[str], embedded_points: np.ndarray, n_clusters: int = 5):
        """
        Initialize the cluster analyzer.

        Args:
            texts: Original text messages
            embedded_points: 2D points from t-SNE or PCA
            n_clusters: Number of clusters to analyze
        """
        self.texts = texts
        self.embedded_points = embedded_points
        self.n_clusters = n_clusters

        # Perform clustering on the 2D points
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(embedded_points)

        # Create TF-IDF vectorizer with specific settings for analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # Include both single words and pairs
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_cluster_characteristics(self) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
        """Analyze what makes each cluster unique."""
        cluster_info = {}

        for cluster_id in range(self.n_clusters):
            # Get texts in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_texts = [t for t, m in zip(self.texts, cluster_mask) if m]

            # Get average TF-IDF scores for this cluster
            cluster_tfidf = self.tfidf_matrix[cluster_mask].toarray()
            avg_tfidf = cluster_tfidf.mean(axis=0)

            # Get top terms by TF-IDF
            top_terms_idx = avg_tfidf.argsort()[-10:][::-1]
            top_terms = [(self.feature_names[i], avg_tfidf[i])
                         for i in top_terms_idx]

            # Get common phrases (more than one word)
            phrases = [(term, score) for term, score in top_terms
                       if ' ' in term]

            # Get message length statistics
            lengths = [len(t.split()) for t in cluster_texts]
            avg_length = np.mean(lengths)

            # Sample representative messages (closest to cluster center)
            center_dists = np.linalg.norm(
                self.embedded_points[cluster_mask] - self.kmeans.cluster_centers_[cluster_id],
                axis=1
            )
            representative_idx = center_dists.argsort()[:3]
            representative_texts = [cluster_texts[i] for i in representative_idx]

            cluster_info[cluster_id] = {
                'size': sum(cluster_mask),
                'top_terms': top_terms,
                'common_phrases': phrases,
                'avg_message_length': avg_length,
                'representative_messages': representative_texts
            }

        return cluster_info

    def print_cluster_summary(self):
        """Print a human-readable summary of each cluster."""
        cluster_info = self.get_cluster_characteristics()

        for cluster_id, info in cluster_info.items():
            print(f"\n=== Cluster {cluster_id} ===")
            print(f"Size: {info['size']} messages")
            print(f"Average message length: {info['avg_message_length']:.1f} words")

            print("\nTop terms/phrases:")
            for term, score in info['top_terms']:
                print(f"  • {term}: {score:.3f}")

            if info['common_phrases']:
                print("\nCommon phrases:")
                for phrase, score in info['common_phrases']:
                    print(f"  • {phrase}: {score:.3f}")

            print("\nRepresentative messages:")
            for i, msg in enumerate(info['representative_messages'], 1):
                print(f"  {i}. {msg[:100]}...")


# Example usage (add to your existing code):
def analyze_clusters(texts: List[str],
                     embedded_points: np.ndarray,
                     output_path: Path = None):
    """
    Analyze and visualize cluster contents.

    Args:
        texts: List of original messages
        embedded_points: 2D points from dimensionality reduction
        output_path: Optional path to save analysis results
    """
    analyzer = ClusterAnalyzer(texts, embedded_points)
    analyzer.print_cluster_summary()

    # Example of how to use with your existing code:
    """
    dim_reduction_data = process_text_for_viz(texts, authors)
    analyze_clusters(texts, dim_reduction_data.embedded_data)
    """