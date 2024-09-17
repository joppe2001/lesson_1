import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('whatsapp-20240910-221731.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the date as the index
df.set_index('timestamp', inplace=True)

# Count messages by author and resample by day
message_counts = df.groupby('author').resample('D').size().unstack(level=0).fillna(0)

# Create the plot
plt.figure(figsize=(12, 6))
message_counts.plot(kind='area', stacked=True)
plt.title('Message Count by Author Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Messages')
plt.legend(title='Author')
plt.tight_layout()
plt.show()