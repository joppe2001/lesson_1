import os
import re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import emoji
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'whatsapp-20240930-201745.csv')


def load_data(file_path):
    """Load and preprocess the WhatsApp chat data."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def preprocess_text(text):
    """Convert text to lowercase and split on whitespace."""
    return re.findall(r'\w+', str(text).lower())


def get_sentiment(text):
    """Determine the sentiment of a given text."""
    sentiment = TextBlob(str(text)).sentiment.polarity
    if sentiment > 0.05:
        return 1  # Positive
    elif sentiment < -0.05:
        return -1  # Negative
    else:
        return 0  # Neutral


def extract_emojis(text):
    """Extract emojis from the given text."""
    return ''.join(c for c in str(text) if c in emoji.EMOJI_DATA)


def enrich_dataframe(df):
    """Add additional features to the dataframe."""
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Text-based features
    df['word_count'] = df['message'].apply(lambda x: len(preprocess_text(x)))
    df['character_count'] = df['message'].str.len()
    df['sentiment'] = df['message'].apply(get_sentiment)
    df['emoji_count'] = df['message'].apply(lambda x: len(extract_emojis(x)))
    df['contains_question'] = df['message'].str.contains('\?').astype(int)
    df['contains_exclamation'] = df['message'].str.contains('!').astype(int)

    # Author-based features
    author_message_counts = df['author'].value_counts()
    df['author_message_count'] = df['author'].map(author_message_counts)

    # Normalize numerical columns
    numeric_columns = ['word_count', 'character_count', 'emoji_count', 'author_message_count']
    df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])

    return df


def create_relationship_plot(df, x_variable, y_variable, plot_type, view_type):
    """Create a plot showing the relationship between two variables."""
    if view_type == 'combined':
        if plot_type == 'scatter':
            fig = px.scatter(df, x=x_variable, y=y_variable)
        elif plot_type == 'line':
            fig = px.line(df, x=x_variable, y=y_variable)
        elif plot_type == 'bar':
            grouped_data = df.groupby(x_variable)[y_variable].mean().reset_index()
            fig = px.bar(grouped_data, x=x_variable, y=y_variable)

        fig.update_layout(title=f'Combined: Relationship between {x_variable} and {y_variable}')

    else:  # separate
        fig = make_subplots(rows=1, cols=2, subplot_titles=df['author'].unique())

        for i, author in enumerate(df['author'].unique(), start=1):
            author_data = df[df['author'] == author]

            if plot_type == 'scatter':
                fig.add_trace(
                    go.Scatter(x=author_data[x_variable], y=author_data[y_variable], mode='markers', name=author),
                    row=1, col=i)
            elif plot_type == 'line':
                fig.add_trace(
                    go.Scatter(x=author_data[x_variable], y=author_data[y_variable], mode='lines', name=author), row=1,
                    col=i)
            elif plot_type == 'bar':
                grouped_data = author_data.groupby(x_variable)[y_variable].mean().reset_index()
                fig.add_trace(go.Bar(x=grouped_data[x_variable], y=grouped_data[y_variable], name=author), row=1, col=i)

        fig.update_layout(title=f'Separate: Relationship between {x_variable} and {y_variable}')

    fig.update_layout(height=600, width=1000)
    return fig


def show_relations(df):
    """Display the relationship exploration section in the Streamlit app."""
    st.header('Explore Relationships')

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_columns) < 2:
        st.warning("Not enough numeric columns to create a relationship plot.")
        return

    x_variable = st.selectbox('Select X-axis variable', numeric_columns)
    y_variable = st.selectbox('Select Y-axis variable', [col for col in numeric_columns if col != x_variable])
    plot_type = st.radio('Select plot type', ['scatter', 'line', 'bar'])
    view_type = st.radio('Select view type', ['combined', 'separate'])

    st.plotly_chart(create_relationship_plot(df, x_variable, y_variable, plot_type, view_type))


def main():
    st.title('WhatsApp Chat Analysis')

    # Load and preprocess data
    df = load_data(DATA_PATH)
    df = enrich_dataframe(df)

    # Display basic statistics
    st.header('Basic Statistics')
    st.write(f"Total Messages: {len(df)}")
    st.write(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    st.write(f"Number of Participants: {df['author'].nunique()}")

    # Show relationship exploration section
    show_relations(df)


if __name__ == "__main__":
    main()