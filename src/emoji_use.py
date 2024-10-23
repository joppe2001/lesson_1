from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from base_plotter import BasePlotter
from typing import Tuple, Dict, Optional, List


@dataclass
class ColumnConfig:
    """Configuration for DataFrame column names"""
    author: str = 'author'
    has_emoji: str = 'has_emoji'


@dataclass
class ChartConfig:
    """Configuration for chart appearance and settings"""
    title: str = 'Emoji Usage by Author'
    xlabel: str = 'Author'
    ylabel: str = 'Percentage'
    figure_size: Tuple[int, int] = (12, 6)
    annotation_settings: Dict = None

    def __post_init__(self):
        self.annotation_settings = self.annotation_settings or {
            'percentage': {
                'fontweight': 'bold',
                'va': 'bottom',
                'ha': 'center'
            },
            'count': {
                'size': 10,
                'va': 'top',
                'ha': 'center'
            }
        }


@dataclass
class OutputConfig:
    """Configuration for output formatting"""
    round_digits: int = 1
    print_summary: bool = True
    save_chart: bool = True


@dataclass
class EmojiStats:
    """Container for emoji statistics results"""
    total_messages: int
    emoji_messages: int
    percentage: float
    by_author: pd.Series


class EmojiAnalyzer:
    """Analyze emoji usage in WhatsApp messages"""

    def __init__(self,
                 columns: Optional[ColumnConfig] = None,
                 chart_config: Optional[ChartConfig] = None,
                 output_config: Optional[OutputConfig] = None,
                 plotter: Optional[BasePlotter] = None):
        """
        Initialize analyzer with custom configurations.

        Example:
            analyzer = EmojiAnalyzer(
                chart_config=ChartConfig(
                    title='Emoji Distribution',
                    figure_size=(15, 8)
                )
            )
        """
        self.cols = columns or ColumnConfig()
        self.chart_config = chart_config or ChartConfig()
        self.output_config = output_config or OutputConfig()
        self.plotter = plotter or BasePlotter(preset='minimal')

    def calculate_statistics(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate basic emoji usage statistics"""
        total_by_author = df.groupby(self.cols.author).size()
        emoji_by_author = df[df[self.cols.has_emoji]].groupby(self.cols.author).size()
        percentage_by_author = (emoji_by_author / total_by_author * 100).round(
            self.output_config.round_digits
        )

        return total_by_author, emoji_by_author, percentage_by_author

    def create_stats_summary(self, df: pd.DataFrame, emoji_by_author: pd.Series) -> EmojiStats:
        """Create summary statistics object"""
        total_emoji_messages = emoji_by_author.sum()
        overall_percentage = (total_emoji_messages / len(df) * 100).round(
            self.output_config.round_digits
        )

        return EmojiStats(
            total_messages=len(df),
            emoji_messages=total_emoji_messages,
            percentage=overall_percentage,
            by_author=emoji_by_author
        )

    def prepare_chart_data(self, percentage_by_author: pd.Series) -> pd.DataFrame:
        """Prepare data for visualization"""
        return pd.DataFrame({
            'Author': percentage_by_author.index,
            'Percentage': percentage_by_author.values
        })

    def add_annotations(self,
                        ax: plt.Axes,
                        percentage_by_author: pd.Series,
                        total_by_author: pd.Series,
                        emoji_by_author: pd.Series) -> None:
        """Add percentage and count annotations to bars"""
        bars = ax.containers[0]
        for i, bar in enumerate(bars):
            author = percentage_by_author.index[i]
            height = bar.get_height()

            # Add percentage on top
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height}%',
                **self.chart_config.annotation_settings['percentage']
            )

            # Add message count below
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                -3,
                f'{emoji_by_author[author]} of {total_by_author[author]} messages',
                **self.chart_config.annotation_settings['count']
            )

    def customize_layout(self, ax: plt.Axes) -> None:
        """Apply custom layout settings"""
        ax.margins(y=0.2)
        plt.subplots_adjust(bottom=0.2)

    def analyze(self, df: pd.DataFrame, output_path: str) -> EmojiStats:
        """
        Run complete emoji analysis pipeline.

        Args:
            df: DataFrame containing message data
            output_path: Path to save visualization

        Returns:
            EmojiStats object containing analysis results
        """
        # Calculate statistics
        total_by_author, emoji_by_author, percentage_by_author = self.calculate_statistics(df)
        stats = self.create_stats_summary(df, emoji_by_author)

        # Create visualization
        if self.output_config.save_chart:
            chart_data = self.prepare_chart_data(percentage_by_author)

            self.plotter.create_barchart(
                data=chart_data,
                title=self.chart_config.title,
                xlabel=self.chart_config.xlabel,
                ylabel=self.chart_config.ylabel
            )

            ax = plt.gca()
            self.add_annotations(ax, percentage_by_author, total_by_author, emoji_by_author)
            self.customize_layout(ax)
            self.plotter.save_plot(output_path)

        # Print summary if enabled
        if self.output_config.print_summary:
            print(f"Total messages with emojis: {stats.emoji_messages}")
            print(f"Percentage of messages with emojis: {stats.percentage:.1f}%")

        return stats