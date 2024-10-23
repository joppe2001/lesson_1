from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from base_plotter import BasePlotter
from typing import Tuple, Dict


@dataclass
class EmojiStats:
    total_messages: int
    emoji_messages: int
    percentage: float
    by_author: pd.Series


def calculate_emoji_statistics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    total_by_author = df.groupby('author').size()
    emoji_by_author = df[df['has_emoji']].groupby('author').size()
    percentage_by_author = (emoji_by_author / total_by_author * 100).round(1)

    return total_by_author, emoji_by_author, percentage_by_author


def create_emoji_stats(df: pd.DataFrame, emoji_by_author: pd.Series) -> EmojiStats:
    total_emoji_messages = emoji_by_author.sum()
    overall_percentage = (total_emoji_messages / len(df) * 100)

    return EmojiStats(
        total_messages=len(df),
        emoji_messages=total_emoji_messages,
        percentage=overall_percentage,
        by_author=emoji_by_author
    )


def prepare_chart_data(percentage_by_author: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'Author': percentage_by_author.index,
        'Percentage': percentage_by_author.values
    })


def add_bar_annotations(ax: plt.Axes,
                        percentage_by_author: pd.Series,
                        total_by_author: pd.Series,
                        emoji_by_author: pd.Series) -> None:
    bars = ax.containers[0]
    for i, bar in enumerate(bars):
        author = percentage_by_author.index[i]
        height = bar.get_height()

        # Add percentage on top
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

        # Add message count below
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            -3,
            f'{emoji_by_author[author]} of {total_by_author[author]} messages',
            ha='center',
            va='top',
            size=10
        )


def customize_chart_layout(ax: plt.Axes) -> None:
    ax.margins(y=0.2)
    plt.subplots_adjust(bottom=0.2)


def emoji_usage_chart(df: pd.DataFrame, output_path: str) -> EmojiStats:
    # Calculate statistics
    total_by_author, emoji_by_author, percentage_by_author = calculate_emoji_statistics(df)

    # Create stats object
    stats = create_emoji_stats(df, emoji_by_author)

    # Prepare chart data
    chart_data = prepare_chart_data(percentage_by_author)

    # Create and customize chart
    plotter = BasePlotter(preset='minimal', figure_size=(12, 6))
    plotter.create_barchart(
        data=chart_data,
        title='Emoji Usage by Author',
        xlabel='Author',
        ylabel='Percentage'
    )

    # Add custom annotations and layout
    ax = plt.gca()
    add_bar_annotations(ax, percentage_by_author, total_by_author, emoji_by_author)
    customize_chart_layout(ax)

    # Save the plot
    plotter.save_plot(output_path)

    # Print summary
    print(f"Total messages with emojis: {stats.emoji_messages}")
    print(f"Percentage of messages with emojis: {stats.percentage:.1f}%")

    return stats