import pandas as pd
import click
from pathlib import Path
from config_handler import ConfigHandler
from visualization import create_simple_message_frequency_plot
from emoji_use import EmojiAnalyzer, ChartConfig, OutputConfig
from timestamp import visualize_hourly_activity
from dataclasses import dataclass
from typing import Optional
from distribution import SentimentAnalyzer, BasePlotter, ColumnConfig
from dimensionality import process_text_for_viz  # New import


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
    """Generate emoji usage visualization with custom settings."""
    # Initialize plotter with desired style
    plotter = BasePlotter(
        preset='minimal',
        figure_size=(12, 8),
        style='seaborn-v0_8-whitegrid'
    )

    # Custom chart configuration
    chart_config = ChartConfig(
        title='Emoji Usage in Chat Messages',
        xlabel='Chat Participant',
        ylabel='Emoji Usage (%)',
        figure_size=(14, 7),
        annotation_settings={
            'percentage': {
                'fontweight': 'bold',
                'va': 'bottom',
                'ha': 'center',
                'fontsize': 11
            },
            'count': {
                'size': 9,
                'va': 'top',
                'ha': 'center',
                'color': 'gray'
            }
        }
    )

    # Custom output configuration
    output_config = OutputConfig(
        round_digits=1,
        print_summary=True,
        save_chart=True
    )

    # Initialize analyzer with configurations
    analyzer = EmojiAnalyzer(
        chart_config=chart_config,
        output_config=output_config,
        plotter=plotter
    )

    # Set output path and run analysis
    output_path = image_dir / 'emoji_usage.jpg'
    stats = analyzer.analyze(df, str(output_path))

    # Print additional insights if desired
    click.echo(f"Emoji usage chart saved as {output_path}")
    click.echo(f"Total messages analyzed: {stats.total_messages}")
    click.echo(f"Messages with emojis: {stats.emoji_messages} ({stats.percentage:.1f}%)")

    return stats

def create_hourly_activity(df, image_dir):
    """Generate hourly activity visualization."""
    output_path = image_dir / 'hourly_activity.jpg'
    visualize_hourly_activity(df, str(output_path))
    click.echo(f"Hourly activity chart saved as {output_path}")


def create_sentiment_analysis(df, image_dir):
    """Generate sentiment analysis visualizations."""
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

    sentiment_dir = image_dir / 'sentiment'
    sentiment_dir.mkdir(exist_ok=True)

    results = analyzer.analyze(df, str(sentiment_dir))
    click.echo(f"Sentiment analysis visualizations saved in {sentiment_dir}")
    return results


def create_text_clusters(df, image_dir):
    """Generate text clustering visualizations using dimension reduction."""
    dim_red_dir = image_dir / 'text_clusters'
    dim_red_dir.mkdir(exist_ok=True)

    click.echo("Starting text clustering analysis...")
    click.echo("Note: This may take several minutes for large datasets.")

    # Initialize plotter
    plotter = BasePlotter(preset='minimal', figure_size=(10, 8))

    # Process with t-SNE first (most time-consuming)
    with click.progressbar(
            length=1,
            label='Generating t-SNE visualization'
    ) as bar:
        click.echo("\nAnalyzing t-SNE clusters...")
        tsne_data = process_text_for_viz(
            texts=df['message'].tolist(),
            authors=df['author'].tolist(),
            method='tsne',
            analyze_clusters=True,  # Add this parameter
            sample_size=60000
        )

        plotter.create_dim_reduction_plot(
            data=tsne_data,
            title='Message Clustering by t-SNE (Sampled Data)',
            output_path=str(dim_red_dir / 'tsne_clusters.jpg')
        )
        bar.update(1)

    # Process with PCA (much faster)
    with click.progressbar(
            length=1,
            label='Generating PCA visualization'
    ) as bar:
        click.echo("\nAnalyzing PCA clusters...")
        pca_data = process_text_for_viz(
            texts=df['message'].tolist(),
            authors=df['author'].tolist(),
            method='pca',
            analyze_clusters=True  # Add this parameter
        )

        plotter.create_dim_reduction_plot(
            data=pca_data,
            title='Message Clustering by PCA (Sampled Data)',
            output_path=str(dim_red_dir / 'pca_clusters.jpg')
        )
        bar.update(1)

    click.echo(f"Text clustering visualizations saved in {dim_red_dir}")


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
        4: ("Sentiment Analysis", create_sentiment_analysis),
        5: ("Text Clustering", create_text_clusters),  # New option
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
        "Please select a visualization (1-5)",  # Updated range
        type=click.IntRange(1, len(visualizations))
    )

    if choice in visualizations:
        click.echo(f"\nGenerating {visualizations[choice][0]}...")
        visualizations[choice][1](df, image_dir)
        click.echo("Visualization completed!")
    else:
        click.echo("Invalid choice. Please select a number between 1 and 5.")


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