# WhatsApp Chat Analyzer

A tool to analyze and visualize WhatsApp chat data, providing insights into your conversations.

## Prerequisites

- Python 3.11
- Either Rye or pip for package management

## Installation

### Using Rye (Recommended)

1. Install Rye:
```bash
# On macOS
brew install rye
```

2. Clone and set up the project:
```bash
# Initialize rye in the project directory
rye sync
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

1. Export your WhatsApp chat (without media) from the WhatsApp app
2. Place the exported `_chat.txt` file in the `data/raw/` directory

### 2. Process the Data

Run the analyzer to convert your raw chat file into a structured format:
```bash
# If using Rye
rye run analyzer

# If using pip
python -m analyzer
```
This will create a processed file in `data/processed/` with a timestamp.

### 3. Clean the Data

1. Open `/notebooks/cleaning/01-cleaning.ipynb`
2. Update the input file path to match your processed data file:
   ```python
   # Change this line to match your processed file name
   datafile = processed / "whatsapp-YYYYMMDD-HHMMSS.csv"
   ```
3. Run all cells in the notebook
4. The processed data will be saved as a Parquet file in the `data/processed/` directory

### 4. Configure Visualization

Update the `config.toml` file to point to your newly created Parquet file:
```toml
[paths]
current = "whatsapp-YYYYMMDD-HHMMSS.parq"
```

### 5. Generate Visualizations

```bash
# Generate basic visualizations
python src/main.py visualize

# Generate all available visualizations
python src/main.py visualize --all
```

Visualizations will be saved in the `visualizations/` directory.

## Output

The analysis will generate several visualizations including:
- Message frequency over time
- Active hours analysis
- User participation statistics
- And more...

## Folder Structure
```
├── data/
│   ├── raw/          # Place your _chat.txt here
│   └── processed/    # CSV files from analyzer
├── notebooks/
│   └── cleaning/     # Data cleaning notebooks
├── src/              # Source code
├── images/           # Output images
└── config.toml       # Configuration file
```

## Troubleshooting

If you encounter NumPy-related warnings, try updating your dependencies:
```bash
# If using Rye
rye sync

# If using pip
pip install e .
```