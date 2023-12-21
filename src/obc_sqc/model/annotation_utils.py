from __future__ import annotations
import pandas as pd
from datetime import timedelta


class AnnotationUtils:  # noqa: D101
    @staticmethod
    def text_annotation(fnl_df, time_window_constant, time_window_constant_max):
        """This algo corresponds the arithmetic with a text annotation.
        We do not include ann_no_datum annotation, because we finally want to show an annotation in minute level
        only when the minute average was not calculated. So, this is annotated by def update_annotation
        If more than one annotation are available for one timeslot, all text annotations are presented.

        fnl_df (df): the resulting df from the "raw_data_suspicious_check" def
        time_window_constant (int): the time window for rolling window checking constant values within it [in minutes]

        result (df): the input df with an extra column "annotation" with text description of annotation
        """  # noqa: D202, D205

        # Create boolean masks for each condition
        mask_obc = fnl_df["ann_obc"] > 0
        mask_invalid_datum = fnl_df["ann_invalid_datum"] > 0
        mask_no_median = fnl_df["ann_no_median"] > 0
        mask_no_datum = fnl_df["ann_no_datum"] > 0
        mask_constant = fnl_df["ann_constant"] > 0
        mask_constant_max = fnl_df["ann_constant_long"] > 0
        mask_constant_frozen = fnl_df["ann_constant_frozen"] > 0

        # Assign corresponding text to each mask
        text_obc = "OBC"
        text_invalid_datum = "SPIKES_INST"
        text_no_median = "NO_MEDIAN"
        text_no_datum = "NO_DATA"
        text_constant = "SHORT_CONST"
        text_constant_max = "LONG_CONST"
        text_constant_frozen = "FROZEN_SENSOR"

        # Concatenate text for all true conditions
        delim: str = ","
        annotation = []
        for i in range(len(fnl_df)):
            text = ""
            if mask_obc[i]:
                text += text_obc + delim
            if mask_invalid_datum[i]:
                text += text_invalid_datum + delim
            if mask_no_median[i]:
                text += text_no_median + delim
            if mask_no_datum[i]:
                text += text_no_datum + delim
            if mask_constant[i]:
                text += text_constant + delim
            if mask_constant_max[i]:
                text += text_constant_max + delim
            if mask_constant_frozen[i]:
                text += text_constant_frozen + delim
            if text.endswith(delim):
                text = text[:-1]

            annotation.append(text)

        fnl_df["annotation"] = annotation

        return fnl_df

    @staticmethod
    def update_ann_text(row, new_text, ann_column):
        """This def passes into the annotation column the given new_text annotation,
        when the value in ann_column is not 0.

        new_text (str): the annotation text
        ann_column (str): the annotation column that needs to be updated

        result (float): the vector average wind speed
        """  # noqa: D202, D205

        if row[f"{ann_column}"] > 0 and new_text not in row["annotation"]:
            if len(row["annotation"]) > 0:
                row["annotation"] += f",{new_text}"
            else:
                row["annotation"] = new_text
        return row

    @staticmethod
    def create_annotations_percentages_list(
        df: pd.DataFrame,
        selected_columns: dict[str, str],
        start_time: pd.Timestamp,
    ) -> pd.Series:
        """Calculates the percentages of faulty data for each annotation type.

            Includes these in a list along with the annotation type str, using the
            following format: [[annotation1, percentage1], [annotation2, percentage2], ...].

        Args:
        ----
            df (pd.DataFrame): the DataFrame containing the raw or minute-averaged data
            selected_columns (dict[str, str]): the dictionary containing the raw or minute-averaged
                                                data annotations and their fault codes
            start_time (pd.Timestamp): the first timestamp of the DataFrame

        Returns:
        -------
            pd.Series: a Series of the format: [[annotation1, percentage1], [annotation2, percentage2], ...]
        """
        hours: range = range(24)

        start_hours: list[pd.Timestamp] = [start_time + pd.DateOffset(hours=hour) for hour in hours]

        # end_hours have xx:00:00 format, so < should be applied instead of <=
        end_hours: list[pd.Timestamp] = [start_time + pd.DateOffset(hours=hour + 1) for hour in hours]

        rounded_hours: list[pd.Timestamp] = [
            (start_hour + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
            for start_hour in start_hours
        ]

        def count_rows_in_range(start: pd.Timestamp, end: pd.Timestamp) -> int:
            """Counts the number of rows of df, the index of which is between start and end datetime.

            Args:
            ----
                start (pd.Timestamp): the starting timestamp
                end (pd.Timestamp): the ending timestamp

            Returns:
            -------
                int: the number of rows between start and end datetime
            """
            number_of_rows: int = df[(df.index >= start) & (df.index < end)].shape[0]
            return number_of_rows

        def count_positive_rows_in_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
            """Counts the number of positive values, the index of which is between start and end datetime.

                The number of positive values is calculated for each column of df.

            Args:
            ----
                start (pd.Timestamp): the starting timestamp
                end (pd.Timestamp): the ending timestamp

            Returns:
            -------
                pd.Series: a Series containing the number of positive rows for each column name
            """
            filtered_df: pd.DataFrame = df[(df.index >= start) & (df.index < end)]
            positive_rows_in_filtered_df: pd.Series = (filtered_df > 0).sum()
            return positive_rows_in_filtered_df

        # Apply aggregation functions on each item of the datetime iterables, using the DataFrame df
        total_rows: list[int] = list(map(count_rows_in_range, start_hours, end_hours))
        positive_rows: list[pd.Series] = list(map(count_positive_rows_in_range, start_hours, end_hours))

        percentages: list[pd.Series] = [(a / b) * 100 for a, b in zip(positive_rows, total_rows, strict=True)]

        error_codes_list: list[list] = [
            (
                [
                    [f"{selected_columns[str(column)]}, {percentage:.1f}"]
                    for column, percentage in item.items()
                    if percentage > 0
                ]
                if item.any() > 0
                else []
            )
            for item in percentages
        ]

        error_codes: pd.Series = pd.Series(error_codes_list, index=rounded_hours)

        return error_codes

    @staticmethod
    def error_codes_hourly(fnl_raw_process: pd.DataFrame, minute_averaging: pd.DataFrame) -> pd.Series:
        """Creates a Series containing the error code annotations based on both raw and minute-averaged data.

            Uses the fnl_raw_process and minute_averaging DataFrames to do so.

        Args:
        ----
            fnl_raw_process (pd.DataFrame): the DataFrame containing the raw data
            minute_averaging (pd.DataFrame): the DataFrame containing the minute-averaged data

        Returns:
        -------
            pd.Series: a Series containing two lists, one for the annotations based on the raw data and
                        one for the annotations based on the minute-averaged data. Each row of the Series
                        has the following format:
                        [[[annotation_raw1, percentage_raw1], ...], [[annotation_min1, percentage_min1], ...]]
        """
        fnl_raw_process["utc_datetime"] = pd.to_datetime(fnl_raw_process["utc_datetime"])
        fnl_raw_process = fnl_raw_process.set_index("utc_datetime")

        # Calculate end (latest second of the examined day) and start time (first second of the examined day)
        end_time: pd.Timestamp = fnl_raw_process.index.max(skipna=False).replace(hour=23, minute=59, second=59)
        start_time = end_time.replace(hour=0, minute=0, second=0)

        # Raw columns and their fault codes
        selected_columns: dict[str, str] = {
            "ann_obc": "OBC",
            "ann_invalid_datum": "SPIKES_INST",
            "ann_no_median": "NO_MEDIAN",
            "ann_no_datum": "NO_DATA",
            "ann_constant": "SHORT_CONST",
            "ann_constant_long": "LONG_CONST",
            "ann_constant_frozen": "FROZEN_SENSOR",
        }

        # Filter the raw DataFrame to include only the last 24hours and the selected columns
        filtered_df: pd.DataFrame = fnl_raw_process.loc[start_time:end_time, list(selected_columns.keys())]

        annotations_based_on_raw: pd.Series = AnnotationUtils.create_annotations_percentages_list(
            filtered_df, selected_columns, start_time
        )

        # Minute columns and their fault codes
        selected_columns = {
            "ann_invalid_datum": "ANOMALOUS_INCREASE",
        }

        # Filter the minute-averaged DataFrame to include only the last 24hours and selected columns
        filtered_df = minute_averaging.loc[start_time:end_time, list(selected_columns.keys())]

        annotations_based_on_minutes: pd.Series = AnnotationUtils.create_annotations_percentages_list(
            filtered_df, selected_columns, start_time
        )

        # merge the raw and minute-averaged annotation/percentages as items of an outer list
        annotations = pd.Series(
            [list(pair) for pair in zip(annotations_based_on_raw, annotations_based_on_minutes, strict=True)],
            index=annotations_based_on_raw.index,
        )

        return annotations
