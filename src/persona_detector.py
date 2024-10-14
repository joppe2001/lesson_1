import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.manifold import TSNE
import tomllib


# Function to remove URLs from text
def remove_url(text):
    return re.sub(r'http\S+', '', text)


# TextClustering class (simplified version of the original)
class TextClustering:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))

    def __call__(self, text, k, labels, batch=False, method="tSNE"):
        if batch:
            parts, _ = self.batch_seq(text, k)
        else:
            parts = text

        X = self.vectorizer.fit_transform(parts)
        X = np.asarray(X.todense())

        distance = manhattan_distances(X, X)

        if method == "tSNE":
            model = TSNE(n_components=2, random_state=42)
            transformed = model.fit_transform(distance)
        else:
            raise ValueError("Only tSNE method is supported in this version")

        self.plot_results(transformed, labels)

    def batch_seq(self, text, k):
        longseq = " ".join(text)
        n = int(len(longseq) / k)
        parts = [longseq[i:i + n] for i in range(0, len(longseq), n)]
        return parts, n

    def plot_results(self, transformed, labels):
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=[hash(label) for label in labels])
        plt.legend(handles=scatter.legend_elements()[0], labels=list(set(labels)), title='Author',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
        plt.xticks([])
        plt.yticks([])
        plt.title("Distinct authors in the WhatsApp dataset")
        plt.tight_layout()

        # Save the plot
        plt.savefig('../images/whatsapp_author_clusters.png')
        plt.show()


# Main script
if __name__ == "__main__":
    # Load configuration
    configfile = Path("../config.toml").resolve()
    with configfile.open("rb") as f:
        config = tomllib.load(f)

    # Load WhatsApp data
    datafile = (Path("..") / Path(config["processed"]) / config["current"]).resolve()
    if not datafile.exists():
        logger.warning("Datafile does not exist. First run src/preprocess.py, and check the timestamp!")
    wa_df = pd.read_parquet(datafile)

    # Process WhatsApp data
    authors = list(np.unique(wa_df.author))
    n = 500
    min_parts = 2

    corpus = {}
    for author in authors:
        subset = wa_df[wa_df.author == author].reset_index()
        longseq = " ".join(subset.message)
        parts = [longseq[i:i + n] for i in range(0, len(longseq), n)]
        parts = [remove_url(chunk) for chunk in parts]
        parts = [re.sub(' +', ' ', chunk) for chunk in parts]
        if len(parts) > min_parts:
            corpus[author] = parts

    # Prepare data for clustering
    text = [part for text in corpus.values() for part in text]
    wa_labels = [k for k, v in corpus.items() for _ in range(len(v))]

    # Perform clustering and plot results
    clustering = TextClustering()
    clustering(text=text, k=100, labels=wa_labels, batch=False, method="tSNE")
