import pandas as pd
import plotly.graph_objects as go


def create_simple_message_frequency_plot(df):
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Resample data to daily message counts for each author
    daily_counts = df.set_index('timestamp').groupby('author').resample('D').size().unstack(level=0).fillna(0)

    # Create the figure
    fig = go.Figure()

    # Add a line for each author
    for author in daily_counts.columns:
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts[author],
            mode='lines',
            name=author,
            hovertemplate='Date: %{x}<br>Messages: %{y}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Daily Message Frequency by Author',
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        legend_title='Author',
        height=600,
        width=1000
    )

    return fig


# Content of src/main.py:

import pandas as pd
from visualization import create_simple_message_frequency_plot


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def main():
    data_path = '../data/whatsapp-20240910-221731.csv'
    df = load_data(data_path)

    # Create and show the plot
    fig = create_simple_message_frequency_plot(df)
    fig.show()


if __name__ == "__main__":
    main()