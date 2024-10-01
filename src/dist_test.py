import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os


def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']


def categorize_sentiment(score):
    if score <= -0.05:
        return 'Negative'
    elif score >= 0.05:
        return 'Positive'
    else:
        return 'Neutral'


def analyze_sentiment_distribution(df, img_path):
    # Perform sentiment analysis if not already done
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['message'].apply(analyze_sentiment)

    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

    # Calculate percentages
    sentiment_counts = df['sentiment_category'].value_counts()
    sentiment_percentages = sentiment_counts / len(df) * 100

    # Create visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Sentiment Analysis Results', fontsize=16)

    # 1. Histogram with KDE
    sns.histplot(df['sentiment'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribution of Sentiment Scores')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Count')
    ax1.axvline(-0.05, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(0.05, color='r', linestyle='--', alpha=0.5)
    ax1.text(-0.5, ax1.get_ylim()[1] * 0.9, 'Negative', ha='center', va='center')
    ax1.text(0, ax1.get_ylim()[1] * 0.9, 'Neutral', ha='center', va='center')
    ax1.text(0.5, ax1.get_ylim()[1] * 0.9, 'Positive', ha='center', va='center')

    # 2. Pie chart of sentiment categories
    colors = [ '#99ff99', '#66b3ff', '#ff9999']
    ax2.pie(sentiment_percentages, labels=sentiment_percentages.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Proportion of Sentiment Categories')

    # 3. Bar plot of sentiment categories
    sns.barplot(x=sentiment_percentages.index, y=sentiment_percentages.values, ax=ax3, palette=colors)
    ax3.set_title('Percentage of Sentiment Categories')
    ax3.set_ylabel('Percentage')
    ax3.set_ylim(0, 100)
    for i, v in enumerate(sentiment_percentages.values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.tight_layout()
    analysis_path = os.path.join(img_path, 'sentiment_analysis_results.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()

    return analysis_path, sentiment_percentages


def print_sentiment_statistics(df, sentiment_percentages):
    print("\nSentiment Category Breakdown:")
    for category, percentage in sentiment_percentages.items():
        print(f"{category}: {percentage:.2f}%")

    print("\nDescriptive Statistics of Sentiment Scores:")
    print(df['sentiment'].describe())

    percentiles = [0, 10, 25, 50, 75, 90, 100]
    print("\nPercentiles of Sentiment Scores:")
    for p in percentiles:
        score = np.percentile(df['sentiment'], p)
        print(f"{p}th percentile: {score:.4f}")


def sentiment_distribution_analysis(df, img_path):
    analysis_path, sentiment_percentages = analyze_sentiment_distribution(df, img_path)
    print(f"Detailed sentiment analysis results saved as: {analysis_path}")
    print_sentiment_statistics(df, sentiment_percentages)

# Usage in main script:
# sentiment_distribution_analysis(df, img_path)