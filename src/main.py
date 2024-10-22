import pandas as pd
import click
from pathlib import Path
from config_handler import ConfigHandler
from visualization import create_simple_message_frequency_plot
from emoji_use import emoji_usage_chart
from timestamp import visualize_hourly_activity
from dataclasses import dataclass
from typing import Optional
from distribution import SentimentAnalyzer, BasePlotter, ColumnConfig  # New import


def load_data(file_path):
    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_message_frequency(df, image_dir):
    """Generate message frequency visualization."""
    fig = create_simple_message_frequency_plot(df)
    output_path = image_dir / 'message_frequency.jpg'
    fig.write_image(str(output_path), scale=2)
    click.echo(f"Message frequency plot saved as {output_path}")


def create_emoji_usage(df, image_dir):
    """Generate emoji usage visualization."""
    output_path = image_dir / 'emoji_usage.jpg'
    emoji_usage_chart(df, str(output_path))
    click.echo(f"Emoji usage chart saved as {output_path}")


def create_hourly_activity(df, image_dir):
    """Generate hourly activity visualization."""
    output_path = image_dir / 'hourly_activity.jpg'
    visualize_hourly_activity(df, str(output_path))
    click.echo(f"Hourly activity chart saved as {output_path}")


def create_sentiment_analysis(df, image_dir):
    """Generate sentiment analysis visualizations."""
    # Initialize sentiment analyzer with default plot settings
    plotter = BasePlotter(
        figure_size=(12, 8),
        style='seaborn-v0_8-darkgrid'
    )

    analyzer = SentimentAnalyzer(
        columns=ColumnConfig(
            timestamp='timestamp',
            message='message',
            author='author'
        ),
        plotter=plotter
    )

    # Create sentiment directory
    sentiment_dir = image_dir / 'sentiment'
    sentiment_dir.mkdir(exist_ok=True)

    # Run analysis
    results = analyzer.analyze(df, str(sentiment_dir))

    click.echo(f"Sentiment analysis visualizations saved in {sentiment_dir}")
    return results


@click.group()
def cli():
    """WhatsApp Chat Analysis Tool - Choose a visualization to generate."""
    pass


@cli.command()
@click.option('--all', is_flag=True, help='Generate all visualizations')
def visualize(all):
    """Generate visualizations based on WhatsApp chat data."""
    config = ConfigHandler()
    config.ensure_directories()

    data_path = config.get_processed_file_path()
    df = load_data(data_path)
    image_dir = config.get_image_dir()

    visualizations = {
        1: ("Message Frequency Plot", create_message_frequency),
        2: ("Emoji Usage Chart", create_emoji_usage),
        3: ("Hourly Activity Visualization", create_hourly_activity),
        4: ("Sentiment Analysis", create_sentiment_analysis),  # New option
    }

    if all:
        click.echo("Generating all visualizations...")
        for _, func in visualizations.values():
            func(df, image_dir)
        click.echo("All visualizations completed!")
        return

    click.echo("Available visualizations:")
    for num, (name, _) in visualizations.items():
        click.echo(f"{num}. {name}")

    choice = click.prompt(
        "Please select a visualization (1-4)",
        type=click.IntRange(1, len(visualizations))
    )

    if choice in visualizations:
        click.echo(f"\nGenerating {visualizations[choice][0]}...")
        visualizations[choice][1](df, image_dir)
        click.echo("Visualization completed!")
    else:
        click.echo("Invalid choice. Please select a number between 1 and 4.")


@cli.command()
def info():
    """Display information about the current configuration."""
    config = ConfigHandler()
    click.echo("\nConfiguration Information:")
    click.echo(f"Data file: {config.get_processed_file_path()}")
    click.echo(f"Images directory: {config.get_image_dir()}")


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed sentiment statistics')
def sentiment(detailed):
    """Analyze sentiment in chat messages."""
    config = ConfigHandler()
    config.ensure_directories()

    data_path = config.get_processed_file_path()
    df = load_data(data_path)
    image_dir = config.get_image_dir()

    click.echo("Running sentiment analysis...")
    results = create_sentiment_analysis(df, image_dir)

    if detailed:
        click.echo("\nDetailed Sentiment Statistics:")
        stats = results.groupby('author')['sentiment'].agg(['mean', 'std', 'count'])
        click.echo(stats.round(3).to_string())


if __name__ == "__main__":
    cli()