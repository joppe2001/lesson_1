import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the data
df = pd.read_csv('whatsapp-20240910-221731.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate message length
df['message_length'] = df['message'].str.len()

# Extract hour and day of week
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Create a pivot table for average message length
heatmap_data = df.pivot_table(values='message_length', index='day_of_week', columns='hour', aggfunc='mean')

# Day names for y-axis
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create the figure
fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Message Length", "Message Length Distribution"),
                    column_widths=[0.7, 0.3])

# Add heatmap
heatmap = go.Heatmap(
    z=heatmap_data.values,
    x=list(range(24)),
    y=days,
    hoverongaps = False,
    hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Avg Length: %{z:.1f} characters<extra></extra>',
    colorscale='RdYlBu_r'
)
fig.add_trace(heatmap, row=1, col=1)

# Add histogram
histogram = go.Histogram(
    x=df['message_length'],
    nbinsx=30,
    hovertemplate='Message Length: %{x} characters<br>Count: %{y}<extra></extra>',
    marker_color='#1f77b4'
)
fig.add_trace(histogram, row=1, col=2)

# Update layout
fig.update_layout(
    title={
        'text': 'WhatsApp Message Length Analysis',
        'font': {'size': 24}
    },
    height=600,
    width=1200,
    showlegend=False
)

# Update x-axis and y-axis properties
fig.update_xaxes(title_text="Hour of Day", row=1, col=1, tickmode='linear', tick0=0, dtick=3)
fig.update_yaxes(title_text="Day of Week", row=1, col=1)
fig.update_xaxes(title_text="Message Length (characters)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

# Update colorbar
fig.update_coloraxes(colorbar_title='Avg Length<br>(characters)', colorbar_len=0.9)

# Add annotations
fig.add_annotation(
    text="Longer messages",
    x=1, y=1, xref="paper", yref="paper",
    showarrow=False,
    font=dict(size=14, color="darkred")
)
fig.add_annotation(
    text="Shorter messages",
    x=1, y=0, xref="paper", yref="paper",
    showarrow=False,
    font=dict(size=14, color="darkblue")
)

# Show the plot
fig.show()

# If you want to save the plot as an HTML file, uncomment the next line
# fig.write_html("message_length_analysis.html")