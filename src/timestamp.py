import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_hourly_activity(df, output_path='whatsapp_hourly_activity.png'):
    """
    Create a visualization of WhatsApp message frequency by hour of the day.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the WhatsApp chat data
    output_path (str): Path to save the output image

    Returns:
    None
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour

    hourly_counts = df['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['hour', 'count']

    plt.figure(figsize=(14, 8))

    sns.barplot(x='hour', y='count', data=hourly_counts, color='#3498db')

    plt.title('WhatsApp Message Frequency by Hour of Day', fontsize=20, pad=20)
    plt.xlabel('Hour of Day (24-hour format)', fontsize=14, labelpad=10)
    plt.ylabel('Number of Messages', fontsize=14, labelpad=10)
    plt.xticks(range(0, 24), fontsize=12)
    plt.yticks(fontsize=12)

    for i, v in enumerate(hourly_counts['count']):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.text(0.5, -0.15,
             f'Time period: {df["timestamp"].min().strftime("%Y-%m-%d")} to {df["timestamp"].max().strftime("%Y-%m-%d")}',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, alpha=0.7)

    peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax(), 'hour']
    plt.axvline(x=peak_hour, color='#e74c3c', linestyle='--', linewidth=2)
    plt.text(peak_hour, plt.ylim()[1], f'Peak Hour: {peak_hour:02d}:00',
             ha='center', va='bottom', color='#e74c3c', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved as '{output_path}'")