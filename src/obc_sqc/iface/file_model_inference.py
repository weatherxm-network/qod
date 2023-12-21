from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time

import pandas as pd

from obc_sqc.model.obc_sqc_driver import ObcSqcCheck
from obc_sqc.schema.schema import SchemaDefinitions

logger = logging.getLogger("obc_sqc")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

import warnings

warnings.filterwarnings("ignore")


def main():
    """The algo requires an input a timeseries in csv with raw data of parameters of 'temperature',
    'humidity', 'wind_speed', 'wind_direction', 'pressure' and 'illuminance'. The following are conducted:
     - create a new timeframe with fixed time interval
     - check for constant data
     - check for jumps and availability in raw level (a)
     - check for jumps and availability in minute level (b)
     - check for availability in hourly level (c)
     - export results from a, b and c into csvs
    """  # noqa: D202, D205, D400, D415

    time.time()

    parser = argparse.ArgumentParser(description="OBC SQC Direct Inference")

    parser.add_argument("--device_id", help="Device ID", required=True)
    parser.add_argument("--date", help="", required=True)
    parser.add_argument("--day1", help="", required=True)
    parser.add_argument("--day2", help="", required=True)
    parser.add_argument("--output_file_path", help="", default="output.parquet")

    (k_args, unknown_args) = parser.parse_known_args(sys.argv[1:])

    args = vars(k_args)

    # Convert start and end dates to datetime
    input_date: datetime = datetime.datetime.strptime(args["date"], "%Y-%m-%d")
    starting_date = input_date - pd.Timedelta(hours=6)
    end_date = input_date + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # QoD object/model/classifier
    qod_model = ObcSqcCheck()

    df1: pd.DataFrame = pd.read_parquet(args["day1"]).query(f"device_id == '{args['device_id']}'")
    df2: pd.DataFrame = pd.read_parquet(args["day2"]).query(f"device_id == '{args['device_id']}'")
    device_df: pd.DataFrame = (
        pd.concat(
            [
                df1,
                df2,
            ]
        )
        .drop(columns=["device_id"])
        .astype(SchemaDefinitions.qod_input_schema())
        .drop_duplicates()
    )

    # In-memory filtering
    df_with_schema: pd.DataFrame = device_df[
        (device_df["utc_datetime"] >= str(starting_date)) & (device_df["utc_datetime"] <= str(end_date))
    ].reset_index(drop=True)

    result_df: pd.DataFrame = qod_model.run(df_with_schema)
    result_df.to_parquet(f"{args['output_file_path']}.parquet", index=False)


if __name__ == "__main__":
    main()
