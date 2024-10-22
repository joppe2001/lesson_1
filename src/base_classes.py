from dataclasses import dataclass

@dataclass
class Regex:
    phone_number: str = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    date_message_split: str = r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s(.+)'

