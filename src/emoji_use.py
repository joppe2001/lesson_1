from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class EmojiStats:
    """Simple container for emoji usage statistics."""
    total_messages: int
    emoji_messages: int
    percentage: float
    by_author: pd.Series


def emoji_usage_chart(df: pd.DataFrame, output_path: str) -> EmojiStats:
    total_by_author = df.groupby('author').size()
    emoji_by_author = df[df['has_emoji']].groupby('author').size()
    percentage_by_author = (emoji_by_author / total_by_author * 100).round(1)

    # Overall statistics
    total_emoji_messages = emoji_by_author.sum()
    overall_percentage = (total_emoji_messages / len(df) * 100)

    # Create stats object
    stats = EmojiStats(
        total_messages=len(df),
        emoji_messages=total_emoji_messages,
        percentage=overall_percentage,
        by_author=emoji_by_author
    )

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Create bars
    bars = plt.bar(
        percentage_by_author.index,
        percentage_by_author.values,
        color='#2ecc71',  # Fresh green color
        width=0.6
    )

    # Customize the chart
    plt.title('Emoji Usage by Author', pad=20, size=14, fontweight='bold')
    plt.ylabel('Messages with Emojis (%)', size=12)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Add the message count below each bar
    for i, author in enumerate(percentage_by_author.index):
        plt.text(
            i,
            -3,  # Position below the bar
            f'{emoji_by_author[author]} of {total_by_author[author]} messages',
            ha='center',
            va='top',
            size=10
        )

    # Customize grid and appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Adjust layout to prevent cutoff
    plt.margins(y=0.2)
    plt.subplots_adjust(bottom=0.2)

    # Save the chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"Total messages with emojis: {stats.emoji_messages}")
    print(f"Percentage of messages with emojis: {stats.percentage:.1f}%")

    return stats