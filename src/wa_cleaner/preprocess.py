import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
import os
sys.path.append('../../src/wa_cleaner')

import click
import pandas as pd
from loguru import logger

from wa_cleaner.settings import (BaseRegexes, Folders, androidRegexes,
                                  iosRegexes, oldRegexes)

logger.remove()
logger.add("logs/logfile.log", rotation="1 week", level="DEBUG")
logger.add(sys.stderr, level="INFO")

logger.debug(f"Python path: {sys.path}")


class WhatsappPreprocessor:
    def __init__(self, folders: Folders, regexes: BaseRegexes):
        self.folders = folders
        self.regexes = regexes

    def __call__(self):
        records, _ = self.process()
        self.save(records)

    def save(self, records: list[tuple]) -> None:
        df = pd.DataFrame(records, columns=["timestamp", "author", "message"])
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        outfile = self.folders.processed / f"whatsapp-{now}.csv"
        logger.info(f"Writing to {outfile}")
        df.to_csv(outfile, index=False)
        logger.success("Done!")

    def process(self) -> tuple:
        records = []
        datafile = self.folders.raw / self.folders.datafile

        tsreg = r"\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\]"
        authorreg = r"] (.+?):"
        fmt = "%d/%m/%Y, %H:%M:%S"

        with datafile.open(encoding="utf-8") as f:
            for line in f.readlines():
                ts = re.search(tsreg, line)
                if ts:
                    timestamp = datetime.strptime(ts.group(1), fmt)
                    author = re.search(authorreg, line)
                    if author:
                        name = author.group(1)
                    else:
                        name = "Unknown"
                    msg = re.sub(tsreg, "", line)
                    msg = re.sub(authorreg, "", msg)
                    records.append((timestamp, name, msg.strip()))
                else:
                    if records:  # Check if records is not empty
                        last_record = records[-1]
                        last_timestamp = last_record[0]
                        last_author = last_record[1]
                        last_msg = last_record[2]
                        records[-1] = (last_timestamp, last_author, last_msg + "\n" + line.strip())
                    else:
                        logger.warning("No records found, skipping line")

        logger.info(f"Found {len(records)} records")
        return records, []


@click.command()
@click.option("--device", default="android", help="Device type: iOS or Android")
def main(device: str):
    if device.lower() == "ios":
        logger.info("Using iOS regexes")
        regexes: BaseRegexes = iosRegexes
    elif device.lower() == "old":
        logger.info("Using old version regexes")
        regexes: BaseRegexes = oldRegexes  # type: ignore
    else:
        logger.info("Using Android regexes")
        regexes: BaseRegexes = androidRegexes  # type: ignore

    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        raw = Path(config["raw"])
        processed = Path(config["processed"])
        datafile = Path(config["input"])

    if not (raw / datafile).exists():
        logger.error(f"File {raw / datafile} not found")

    folders = Folders(
        raw=raw,
        processed=processed,
        datafile=datafile,
    )
    preprocessor = WhatsappPreprocessor(folders, regexes)
    preprocessor()


if __name__ == "__main__":
    main()
