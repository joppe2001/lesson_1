import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('whatsapp-20240910-221731.csv')

# Count emoji usage by author
emoji_usage = df[df['has_emoji']].groupby('author').size()

# Create a pie chart
plt.figure(figsize=(10, 6))
plt.pie(emoji_usage, labels=emoji_usage.index, autopct='%1.1f%%', startangle=90)
plt.title('Emoji Usage by Author')
plt.axis('equal')
plt.show()

# Print the total number of messages with emojis
print(f"Total messages with emojis: {emoji_usage.sum()}")
print(f"Percentage of messages with emojis: {(emoji_usage.sum() / len(df) * 100):.2f}%")