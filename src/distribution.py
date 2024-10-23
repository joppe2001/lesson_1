import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from base_plotter import BasePlotter


@dataclass
class ColumnConfig:
    """
    Configuration for DataFrame column names.

    Customize these if your DataFrame columns have different names:
    - timestamp: Column containing message dates/times
    - message: Column containing message text
    - author: Column containing message sender names
    - month: Name for the generated month column
    - sentiment: Name for the generated sentiment score column

    Example:
        custom_cols = ColumnConfig(
            timestamp='date_sent',
            message='text',
            author='sender'
        )
    """
    timestamp: str = 'timestamp'
    message: str = 'message'
    author: str = 'author'
    month: str = 'month'
    sentiment: str = 'sentiment'


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    Customize these to change how visualizations look and where they're saved:
    - titles: Dictionary of plot titles for each visualization type
    - labels: Dictionary of axis labels
    - output_files: Dictionary of output filenames

    Example:
        viz_config = VisualizationConfig(
            titles={'monthly': 'Emotion Trends', 'distribution': 'Mood Distribution'},
            labels={'sentiment': 'Emotional Score'},
            output_files={'monthly': 'emotion_trends.png'}
        )
    """
    titles: Dict[str, str] = None
    labels: Dict[str, str] = None
    output_files: Dict[str, str] = None

    def __post_init__(self):
        # Default visualization settings - override these by passing values to constructor
        self.titles = self.titles or {
            'monthly': 'Monthly Average Sentiment',
            'distribution': 'Distribution of Sentiment Scores',
            'boxplot': 'Sentiment Distribution by Author',
            'heatmap': 'Sentiment Heatmap by Author and Month'
        }

        self.labels = self.labels or {
            'sentiment': 'Sentiment Score (-1 to 1)',
            'sentiment_dist': 'Sentiment Score'
        }

        self.output_files = self.output_files or {
            'monthly': 'monthly_sentiment.png',
            'distribution': 'sentiment_distribution.png',
            'boxplot': 'sentiment_by_author.png',
            'heatmap': 'sentiment_heatmap.png'
        }


@dataclass
class AnalysisConfig:
    """
    Configuration for analysis settings.

    Customize these to change how the analysis is performed:
    - round_digits: Number of decimal places in results
    - summary_metrics: List of pandas aggregation functions to apply
    - summary_columns: Names for the resulting summary columns

    Example:
        analysis_config = AnalysisConfig(
            round_digits=2,
            summary_metrics=['mean', 'count', 'std'],
            summary_columns=['avg_mood', 'num_messages', 'mood_variation']
        )
    """
    round_digits: int = 3
    summary_metrics: List[str] = None
    summary_columns: List[str] = None

    def __post_init__(self):
        # Default analysis settings - override these by passing values to constructor
        self.summary_metrics = self.summary_metrics or ['mean', 'count']
        self.summary_columns = self.summary_columns or [
            'avg_sentiment', 'message_count', 'first_message', 'last_message'
        ]


class SentimentAnalyzer:
    """
    Analyze sentiment in WhatsApp messages.

    Main customization points:
    1. Initialize with custom configs:
       analyzer = SentimentAnalyzer(
           columns=custom_cols,
           viz_config=viz_config,
           analysis_config=analysis_config,
           plotter=custom_plotter
       )

    2. Override individual methods for custom behavior:
       - analyze_sentiment: Change how sentiment is calculated
       - prepare_data: Modify data preparation steps
       - create_user_summary: Change how summary is generated
       - create_visualizations: Add/modify visualizations
    """

    def __init__(self,
                 columns: Optional[ColumnConfig] = None,
                 viz_config: Optional[VisualizationConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None,
                 plotter: Optional[BasePlotter] = None):
        """
        Initialize analyzer with custom configurations.
        All parameters are optional and have sensible defaults.
        """
        self.cols = columns or ColumnConfig()
        self.viz_config = viz_config or VisualizationConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        self.plotter = plotter or BasePlotter(preset='dark')
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for a single message.
        Override this method to use a different sentiment analysis approach.
        """
        return self.analyzer.polarity_scores(text)['compound']

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis.
        Override this method to add more data preparation steps.
        """
        df = df.copy()
        df[self.cols.timestamp] = pd.to_datetime(df[self.cols.timestamp])
        df[self.cols.sentiment] = df[self.cols.message].apply(self.analyze_sentiment)
        df[self.cols.month] = df[self.cols.timestamp].dt.to_period('M')
        return df

    def create_user_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user sentiment summary.
        Override this method to change summary statistics or format.
        """
        metrics = {
            self.cols.sentiment: self.analysis_config.summary_metrics,
            self.cols.timestamp: ['min', 'max']
        }

        summary = df.groupby(self.cols.author).agg(metrics).round(
            self.analysis_config.round_digits
        )
        summary.columns = self.analysis_config.summary_columns
        return summary

    def create_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Create all visualizations.
        Override or modify this method to add/change visualizations.
        Each visualization uses settings from self.viz_config.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Monthly sentiment trend
        monthly_sentiment = df.groupby(self.cols.month)[self.cols.sentiment].mean()
        self.plotter.create_time_series(
            monthly_sentiment,
            self.viz_config.titles['monthly'],
            self.viz_config.labels['sentiment'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['monthly'])
        )

        # 2. Sentiment distribution
        self.plotter.create_distribution(
            df[self.cols.sentiment],
            self.viz_config.titles['distribution'],
            self.viz_config.labels['sentiment_dist'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['distribution'])
        )

        # 3. Sentiment heatmap
        pivot_data = df.pivot_table(
            values=self.cols.sentiment,
            index=self.cols.month,
            columns=self.cols.author,
            aggfunc='mean'
        )
        self.plotter.create_heatmap(
            pivot_data,
            self.viz_config.titles['heatmap'],
            output_path=os.path.join(output_dir, self.viz_config.output_files['heatmap'])
        )

    def analyze(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """
        Main analysis pipeline.
        Orchestrates the complete analysis process.
        Returns the DataFrame with added sentiment scores.
        """
        df = self.prepare_data(df)
        self.create_visualizations(df, output_dir)
        user_summary = self.create_user_summary(df)
        print("\nUser Summary:")
        print(user_summary.sort_values('avg_sentiment', ascending=False))
        return df