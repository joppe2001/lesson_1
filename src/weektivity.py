import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Read the data
df = pd.read_csv('../data/processed/whatsapp-20240910-221731.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp
df = df.sort_values('timestamp')

# Calculate time since last message
df['time_since_last'] = df['timestamp'].diff().dt.total_seconds() / 60  # in minutes

# Calculate message length
df['message_length'] = df['message'].str.len()

# Resample data to reduce density (e.g., hourly averages)
df_resampled = df.set_index('timestamp').resample('H').agg({
    'author': 'first',
    'message_length': 'mean',
    'time_since_last': 'mean'
}).reset_index()

# Create a color map for authors
authors = df['author'].unique()
color_map = {author: f'hsl({i*360/len(authors)},70%,50%)' for i, author in enumerate(authors)}

# Create the figure
fig = make_subplots(rows=1, cols=1, subplot_titles=("Hourly Conversation Dynamics"))

# Conversation Dynamics
for author in authors:
    author_df = df_resampled[df_resampled['author'] == author]
    fig.add_trace(
        go.Scatter(
            x=author_df['timestamp'],
            y=author_df['message_length'],
            mode='markers',
            name=author,
            marker=dict(
                size=author_df['time_since_last'].clip(5, 30),
                color=color_map[author],
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            hovertemplate='Author: %{fullData.name}<br>Time: %{x}<br>Avg Message Length: %{y:.1f}<br>Avg Minutes Since Last: %{marker.size:.1f}<extra></extra>'
        )
    )

# Update layout
fig.update_layout(
    title={
        'text': 'WhatsApp Conversation Dynamics (Hourly Average)',
        'font': {'size': 24}
    },
    height=700,
    width=1200,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest"
)

# Update axes
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Average Message Length (characters)")

# Add annotations
fig.add_annotation(
    text="Larger bubbles indicate longer time since last message",
    x=0.02, y=1.05, xref="paper", yref="paper",
    showarrow=False,
    font=dict(size=12)
)

# Show the plot
fig.show()

# If you want to save the plot as an HTML file, uncomment the next line
# fig.write_html("simplified_conversation_dynamics.html")