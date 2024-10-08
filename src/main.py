import pandas as pd

from relations import show_relations
from visualization import create_simple_message_frequency_plot
from emoji_use import emoji_usage_chart
from timestamp import visualize_hourly_activity
from distribution_vis import analyze_whatsapp_data
from dist_test import sentiment_distribution_analysis
from relations import run_whatsapp_analysis
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(current_dir, '..', 'data', 'processed', 'whatsapp-20240930-201745.csv')

    df = load_data(data_path)

    fig = create_simple_message_frequency_plot(df)
    output_path = os.path.join(current_dir, '..', 'images','message_frequency.jpg')
    fig.write_image(output_path, scale=2)
    print(f"Message frequency plot saved as {output_path}")

    output_path_emoji = os.path.join(current_dir, '..', 'images', 'emoji_usage.jpg')
    emoji_usage_chart(df, output_path_emoji)
    visualize_hourly_activity(
        df,
        os.path.join(current_dir, '..', 'images', 'hourly_activity.jpg'))
    print(f"Emoji usage chart saved as {output_path_emoji}")

    # output_path_dist = os.path.join(current_dir, '..', 'images')
    # sentiment_analysis(df, output_path_dist)
    # print("Analysis complete!")

    # output_path_dist_test = os.path.join(current_dir, '..', 'images')
    # sentiment_distribution_analysis(df, output_path_dist_test)

    run_whatsapp_analysis(df)

    show_relations(df)

if __name__ == "__main__":
    main()