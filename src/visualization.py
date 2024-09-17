import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_simple_message_frequency_plot(df):
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Resample data to weekly message counts for each author
    weekly_counts = df.set_index('timestamp').groupby('author').resample('W').size().unstack(level=0).fillna(0)

    # Calculate total weekly messages
    weekly_counts['Total'] = weekly_counts.sum(axis=1)

    # Create the figure with two subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Weekly Message Count by Author", "Total Weekly Messages"))

    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Add bars for each author
    for i, author in enumerate(weekly_counts.columns[:-1]):  # Exclude 'Total'
        fig.add_trace(
            go.Bar(x=weekly_counts.index, y=weekly_counts[author],
                   name=author, marker_color=colors[i % len(colors)]),
            row=1, col=1
        )

    # Add total messages bar
    fig.add_trace(
        go.Bar(x=weekly_counts.index, y=weekly_counts['Total'],
               name='Total', marker_color='black'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title='WhatsApp Conversation Activity Over Time',
        height=800, width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='stack',
        annotations=[
            dict(text="Key Insights:", x=0, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="black")),
            dict(text="1. Weekly message count comparison between users", x=0, y=1.04, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray")),
            dict(text="2. Overall trend in conversation activity", x=0, y=1.00, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray"))
        ]
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Messages per Week", row=1, col=1)
    fig.update_yaxes(title_text="Total Messages", row=2, col=1)

    return fig

# Main execution
def main():
    data_path = '../data/processed/whatsapp-20240910-221731.csv'
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig = create_simple_message_frequency_plot(df)
    fig.show()

if __name__ == "__main__":
    main()