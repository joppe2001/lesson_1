from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class ChartStyle:
    """Configuration for chart appearance."""
    figure_size: tuple[int, int] = (14, 8)
    bar_color: str = '#3498db'
    peak_line_color: str = '#e74c3c'
    title_size: int = 20
    label_size: int = 14
    tick_size: int = 12
    annotation_size: int = 10
    grid_alpha: float = 0.7
    dpi: int = 300


@dataclass
class ActivityStats:
    """Container for hourly activity statistics."""
    hourly_counts: pd.DataFrame
    peak_hour: int
    start_date: datetime
    end_date: datetime
    total_messages: int


def analyze_hourly_activity(df: pd.DataFrame) -> ActivityStats:
    """Analyze hourly activity patterns in the data."""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

    hourly_counts = df['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['hour', 'count']

    return ActivityStats(
        hourly_counts=hourly_counts,
        peak_hour=hourly_counts.loc[hourly_counts['count'].idxmax(), 'hour'],
        start_date=pd.to_datetime(df['timestamp']).min(),
        end_date=pd.to_datetime(df['timestamp']).max(),
        total_messages=len(df)
    )


def visualize_hourly_activity(df: pd.DataFrame,
                              output_path: str,
                              style: Optional[ChartStyle] = None) -> ActivityStats:
    """
    Create a visualization of WhatsApp message frequency by hour of the day.

    Args:
        df: DataFrame containing WhatsApp chat data with 'timestamp' column
        output_path: Path to save the output image
        style: Optional ChartStyle object for customizing appearance
    """
    style = style or ChartStyle()
    stats = analyze_hourly_activity(df)

    # Create the plot
    plt.figure(figsize=style.figure_size)

    # Create base bar plot
    sns.barplot(
        x='hour',
        y='count',
        data=stats.hourly_counts,
        color=style.bar_color
    )

    # Configure title and labels
    plt.title('WhatsApp Message Frequency by Hour of Day',
              fontsize=style.title_size,
              pad=20)
    plt.xlabel('Hour of Day (24-hour format)',
               fontsize=style.label_size,
               labelpad=10)
    plt.ylabel('Number of Messages',
               fontsize=style.label_size,
               labelpad=10)

    # Configure ticks
    plt.xticks(range(0, 24), fontsize=style.tick_size)
    plt.yticks(fontsize=style.tick_size)

    # Add value labels on bars
    for i, v in enumerate(stats.hourly_counts['count']):
        plt.text(i, v + 0.5, str(v),
                 ha='center',
                 va='bottom',
                 fontsize=style.annotation_size)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=style.grid_alpha)

    # Add time period text
    period_text = (f'Time period: {stats.start_date.strftime("%Y-%m-%d")} '
                   f'to {stats.end_date.strftime("%Y-%m-%d")}')
    plt.text(0.5, -0.15, period_text,
             ha='center',
             va='center',
             transform=plt.gca().transAxes,
             fontsize=style.tick_size,
             alpha=0.7)

    # Add peak hour line
    plt.axvline(x=stats.peak_hour,
                color=style.peak_line_color,
                linestyle='--',
                linewidth=2)
    plt.text(stats.peak_hour,
             plt.ylim()[1],
             f'Peak Hour: {stats.peak_hour:02d}:00',
             ha='center',
             va='bottom',
             color=style.peak_line_color,
             fontsize=style.tick_size,
             fontweight='bold')

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=style.dpi, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved as '{output_path}'")
    return stats