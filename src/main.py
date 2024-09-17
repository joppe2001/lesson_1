import pandas as pd
from visualization import create_simple_message_frequency_plot


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def main():
    data_path = '../data/whatsapp-20240910-221731.csv'
    df = load_data(data_path)

    fig = create_simple_message_frequency_plot(df)

    fig.write_image("message_frequency.pdf")
    print("Plot saved as message_frequency.pdf")


if __name__ == "__main__":
    main()