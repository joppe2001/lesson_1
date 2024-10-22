import pandas as pd
import click
from config_handler import ConfigHandler
from visualization import create_simple_message_frequency_plot
from emoji_use import emoji_usage_chart
from timestamp import visualize_hourly_activity
import os


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

    if all:
        click.echo("Generating all visualizations...")
        create_message_frequency(df, image_dir)
        create_emoji_usage(df, image_dir)
        create_hourly_activity(df, image_dir)
        click.echo("All visualizations completed!")
        return

    visualizations = {
        1: ("Message Frequency Plot", create_message_frequency),
        2: ("Emoji Usage Chart", create_emoji_usage),
        3: ("Hourly Activity Visualization", create_hourly_activity),
    }

    click.echo("Available visualizations:")
    for num, (name, _) in visualizations.items():
        click.echo(f"{num}. {name}")

    choice = click.prompt("Please select a visualization (1-3)", type=int)

    if choice in visualizations:
        click.echo(f"\nGenerating {visualizations[choice][0]}...")
        visualizations[choice][1](df, image_dir)
        click.echo("Visualization completed!")
    else:
        click.echo("Invalid choice. Please select a number between 1 and 3.")


@cli.command()
def info():
    """Display information about the current configuration."""
    config = ConfigHandler()
    click.echo("\nConfiguration Information:")
    click.echo(f"Data file: {config.get_processed_file_path()}")
    click.echo(f"Images directory: {config.get_image_dir()}")


if __name__ == "__main__":
    cli()