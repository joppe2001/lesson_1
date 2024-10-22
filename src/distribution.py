# distribution.py
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wa_cleaner.humanhasher import humanize
import os
from base_plotter import BasePlotter

@dataclass
class ColumnConfig:
    """Configuration for DataFrame column names"""
    timestamp: str = 'timestamp'
    message: str = 'message'
    author: str = 'author'

class SentimentAnalyzer:
    """Analyze sentiment in WhatsApp messages"""

    def __init__(self,
                 columns: Optional[ColumnConfig] = None,
                 plotter: Optional[BasePlotter] = None):
        self.cols = columns or ColumnConfig()
        self.plotter = plotter or BasePlotter(preset='dark')
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> float:
        """Get compound sentiment score for text"""
        return self.analyzer.polarity_scores(text)['compound']

    def analyze(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """Run sentiment analysis and create visualizations"""
        # Prepare data
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df[self.cols.timestamp])
        df['sentiment'] = df[self.cols.message].apply(self.analyze_sentiment)
        df['month'] = df['timestamp'].dt.to_period('M')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Monthly sentiment trend
        monthly_sentiment = df.groupby('month')['sentiment'].mean()
        self.plotter.create_time_series(
            monthly_sentiment,
            'Monthly Average Sentiment',
            'Sentiment Score (-1 to 1)',
            output_path=os.path.join(output_dir, 'monthly_sentiment.png')
        )

        # 2. User sentiment summary
        user_summary = df.groupby(self.cols.author).agg({
            'sentiment': ['mean', 'count'],
            'timestamp': ['min', 'max']
        }).round(3)
        user_summary.columns = ['avg_sentiment', 'message_count', 'first_message', 'last_message']

        print("\nUser Summary:")
        print(user_summary.sort_values('avg_sentiment', ascending=False))

        # 3. Sentiment distribution
        self.plotter.create_distribution(
            df['sentiment'],
            'Distribution of Sentiment Scores',
            'Sentiment Score',
            output_path=os.path.join(output_dir, 'sentiment_distribution.png')
        )

        return df