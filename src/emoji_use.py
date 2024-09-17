import matplotlib.pyplot as plt

def emoji_usage_chart(df, output_path):
    emoji_usage = df[df['has_emoji']].groupby('author').size()

    # Create a pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(emoji_usage, labels=emoji_usage.index, autopct='%1.1f%%', startangle=90)
    plt.title('Emoji Usage by Author')
    plt.axis('equal')

    # Save the chart as an image file
    plt.savefig(output_path)
    plt.close()

    # Print the total number of messages with emojis
    total_emoji_messages = emoji_usage.sum()
    percentage_emoji_messages = (total_emoji_messages / len(df) * 100)
    print(f"Total messages with emojis: {total_emoji_messages}")
    print(f"Percentage of messages with emojis: {percentage_emoji_messages:.2f}%")