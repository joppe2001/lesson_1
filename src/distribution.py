import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wa_cleaner.humanhasher import humanize
import os


def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound'], sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos']


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def sentiment_analysis(df, img_path):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sentiment'], df['neg'], df['neu'], df['pos'] = zip(*df['message'].apply(analyze_sentiment))
    df['month'] = df['timestamp'].dt.to_period('M')

    # Print diagnostic information
    print(f"Total messages: {len(df)}")
    print(f"Messages with negative sentiment: {len(df[df['sentiment'] < 0])}")
    print(f"Messages with positive sentiment: {len(df[df['sentiment'] > 0])}")
    print(f"Messages with neutral sentiment: {len(df[df['sentiment'] == 0])}")
    print(f"Minimum sentiment score: {df['sentiment'].min()}")
    print(f"Maximum sentiment score: {df['sentiment'].max()}")

    # Print a few examples of messages with their sentiment scores
    print("\nExample messages and their sentiment scores:")
    sample_messages = df.sample(n=min(5, len(df)))
    for _, row in sample_messages.iterrows():
        print(f"Message: {row['message'][:50]}...")
        print(
            f"Sentiment: {row['sentiment']:.4f} (neg: {row['neg']:.4f}, neu: {row['neu']:.4f}, pos: {row['pos']:.4f})")
        print()

    users = df['author'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(users)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1.5]})
    fig.suptitle('WhatsApp Conversation Mood Analysis', fontsize=16)

    # Monthly average sentiment
    monthly_sentiment = df.groupby('month')['sentiment'].mean()
    ax1.plot(monthly_sentiment.index.astype(str), monthly_sentiment.values, color='black', linewidth=2)
    ax1.set_title('Monthly Average Mood', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('Sentiment (-1 to 1)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(monthly_sentiment.index.astype(str), monthly_sentiment.values, 0,
                     where=(monthly_sentiment.values > 0), interpolate=True, color='lightgreen', alpha=0.3)
    ax1.fill_between(monthly_sentiment.index.astype(str), monthly_sentiment.values, 0,
                     where=(monthly_sentiment.values <= 0), interpolate=True, color='lightcoral', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # User summary
    user_summary = df.groupby('author').agg({
        'sentiment': ['mean', 'std', 'count'],
        'neg': 'mean',
        'neu': 'mean',
        'pos': 'mean',
        'timestamp': ['min', 'max']
    })
    user_summary.columns = ['avg_sentiment', 'std_sentiment', 'message_count', 'avg_neg', 'avg_neu', 'avg_pos',
                            'first_message', 'last_message']
    user_summary = user_summary.sort_values('avg_sentiment', ascending=True)

    # Enhanced user summary visualization
    bar_height = 0.5
    y_pos = np.arange(len(users))

    # Negative sentiment bars
    ax2.barh(y_pos, user_summary['avg_neg'], height=bar_height, align='center', color='lightcoral', alpha=0.7,
             label='Negative')

    # Neutral sentiment bars
    ax2.barh(y_pos, user_summary['avg_neu'], height=bar_height, align='center', left=user_summary['avg_neg'],
             color='lightgray', alpha=0.7, label='Neutral')

    # Positive sentiment bars
    ax2.barh(y_pos, user_summary['avg_pos'], height=bar_height, align='center',
             left=user_summary['avg_neg'] + user_summary['avg_neu'], color='lightgreen', alpha=0.7, label='Positive')

    # Add average sentiment markers
    for i, (idx, row) in enumerate(user_summary.iterrows()):
        ax2.scatter(row['avg_sentiment'], i, color='black', s=50, zorder=3)
        ax2.plot([row['avg_sentiment'], row['avg_sentiment']], [i - bar_height / 2, i + bar_height / 2], color='black',
                 linewidth=2, zorder=2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([humanize(user) for user in user_summary.index])
    ax2.invert_yaxis()
    ax2.set_xlabel('Sentiment Components and Average')
    ax2.set_title('User Mood Summary', fontsize=14)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')

    # Add message count and date range annotations
    for i, (idx, row) in enumerate(user_summary.iterrows()):
        ax2.annotate(f"Messages: {row['message_count']}", xy=(1.02, i), xycoords=('axes fraction', 'data'),
                     va='center', ha='left', fontsize=8)
        date_range = f"{row['first_message'].strftime('%Y-%m-%d')} to {row['last_message'].strftime('%Y-%m-%d')}"
        ax2.annotate(date_range, xy=(1.02, i - 0.3), xycoords=('axes fraction', 'data'),
                     va='center', ha='left', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(img_path, 'whatsapp_mood_analysis.png')
    ensure_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Mood analysis completed. Visualization saved as: {output_path}")

    # Create a histogram of sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=50, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    hist_path = os.path.join(img_path, 'sentiment_distribution.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sentiment distribution histogram saved as: {hist_path}")