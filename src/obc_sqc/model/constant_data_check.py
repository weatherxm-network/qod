from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Tuple


class ConstantDataCheck:
    """Functions for checking for constant data within a time window."""

    @staticmethod
    def assign_first_value_in_window(window: pd.Series) -> np.float64:
        """Return the first non-nan value present in the window, else 0.

        Args:
        ----
            window (pd.Series): The window containing the values.

        Returns:
        -------
            np.float64: The desired value
        """
        assign_value: np.float64 = window.loc[window.first_valid_index()] if not window.isna().all() else 0

        return assign_value

    @staticmethod
    def get_number_of_rows_of_last_day(fnl_df: pd.DataFrame, time_window_constant: int) -> int:
        """Get the number of rows of the last day of the dataframe, within the time window.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe.
            time_window_constant (int): The time window.

        Returns:
        -------
            int: The calculated number of rows
        """
        if time_window_constant is np.nan:
            return time_window_constant

        fnl_df.index = pd.to_datetime(fnl_df.index)
        last_timestamp: pd.Timestamp = fnl_df.index[-1]
        start_timestamp: pd.Timestamp = last_timestamp - pd.Timedelta(minutes=time_window_constant)

        filtered_df: pd.DataFrame = fnl_df[(fnl_df.index >= start_timestamp) & (fnl_df.index < last_timestamp)]
        number_of_rows: int = len(filtered_df)

        return number_of_rows

    @staticmethod
    def check_constant_humidity_temperature(
        fnl_df: pd.DataFrame, time_window_const: int, ann_constant: int, rh_threshold: float
    ) -> pd.DataFrame:
        """Check for constant values in a small time-window for temperature or humidity data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation that will be used for values identified as constant
            rh_threshold (float): The threshold below which constant humidity values are suspicious

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which a column for constant data annotation is added
        """
        fnl_df.index = pd.to_datetime(fnl_df.index)

        # Create a column for the median in the rolling window
        fnl_df["median"] = (
            fnl_df["humidity_for_raw_check"]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(np.nanmedian)
        )

        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const
        )

        # Filter rows based on conditions
        condition: pd.Series = (
            (fnl_df["non_nan_count"] == time_window_const_as_row_count)  # all values non-nan
            & (fnl_df["unique_values"] == 1)  # a constant value across all elements
            & (fnl_df["median"] < rh_threshold)
        )

        # Hereafter, when a condition is used as condition[::-1], it is because we want to annotate the whole
        # time window (time_window_const_as_row_count), and the annotations should be applied sequentially,
        # over rolling time windows. That means, that for each row, an annotation may be overwritten, and
        # the last one will be kept. To mimic this behaviour in a more effective way, we apply the rolling
        # time window backwards and check the last true condition in the (backward) of the rolling time window.

        # Guard datetime ensures that we check the corresponding condition and apply the annotations only
        # for rows belonging to a date [me_window_const minutes] after the start of the data in our dataframe.
        # This way, if we select, for example, a time window of 24 hours for a dataframe the data of which
        # begin on 18:00 of the previous day, only faulty data after 18:00 of the current day will be taken
        # into consideration to assign an annotation.

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const}min")

        # Look in each time window backwards and get the sum of the values of the condition series in the
        # window (the values are 0 or 1 for False or True respectively). If sum > 0, then at least one
        # True value exists in the window, so the data are annotated.
        annotation_condition_reversed: pd.Series = (
            condition.iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(lambda x: x[x.index >= guard_datetime].sum())
        )

        # annotation_condition_reversed > 0 provides a 0/1 mask and when multiplied with "ann_constant"
        # gives 0 for annotation_condition_reversed=False and ann_constant for
        # annotation_condition_reversed=True
        fnl_df["ann_constant"] = (ann_constant * annotation_condition_reversed > 0).astype(int)[::-1]

        fnl_df = fnl_df.drop(["unique_values", "median"], axis=1)

        return fnl_df

    @staticmethod
    def prepare_wind_df_and_condition(fnl_df: pd.DataFrame, time_window_const: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare the dataframe for wind constant data checking.

        This involves the operations that are common for both wind speed and wind direction
        constant data identification.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which columns for median values are added.
        """
        # Create a column for the median temperature in the rolling window
        fnl_df["median_temperature"] = (
            fnl_df["temperature_for_raw_check"]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(np.nanmedian)
        )

        # Create a column for the median humidity in the rolling window
        #fnl_df["median_humidity"] = (
        #    fnl_df["humidity_for_raw_check"]
        #    .rolling(f"{time_window_const}min", min_periods=1, closed="left")
        #    .apply(np.nanmedian)
        #)
        fnl_df["median_humidity"] = (
            fnl_df["humidity_for_raw_check"]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(lambda x: np.nanpercentile(x, 50))
        )

        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const
        )

        # Filter rows based on conditions
        all_non_nan_constant: pd.Series = (fnl_df["non_nan_count"] == time_window_const_as_row_count) & (
            fnl_df["unique_values"] == 1
        )

        return fnl_df, all_non_nan_constant

    @staticmethod
    def check_constant_wind_direction(
        fnl_df: pd.DataFrame, time_window_const: int, ann_constant: int, ann_constant_frozen: int
    ) -> pd.DataFrame:
        """Check for constant values in a small time window for wind direction data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation that will be used for values identified as constant
            ann_constant_frozen (int): The annotation to be used for values identified as constant
                                        because of a frozen sensor

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which columns for constant data annotation are added
        """
        all_non_nan_constant: pd.Series
        fnl_df, all_non_nan_constant = ConstantDataCheck.prepare_wind_df_and_condition(fnl_df, time_window_const)
        fnl_df.index = pd.to_datetime(fnl_df.index)

        # Filter rows based on conditions
        temperature_lt_0: pd.Series = fnl_df["median_temperature"] <= 0
        temp_gt_0_hum_lt_85: pd.Series = (fnl_df["median_temperature"] > 0) & (
            fnl_df["median_humidity"] < 85  # noqa: PLR2004
        )

        all_non_nan_constant_temperature_lt_0: pd.Series = all_non_nan_constant & temperature_lt_0
        all_non_nan_constant_temp_gt_0_hum_lt_85: pd.Series = all_non_nan_constant & temp_gt_0_hum_lt_85

        # all_non_nan_constant_temperature_lt_0 condition has higher priority in an if-else structure,
        # so it is applied later in the sequence.
        ann_constant_decision: np.ndarray = np.where(
            all_non_nan_constant_temp_gt_0_hum_lt_85, ann_constant, fnl_df["ann_constant"]
        )
        fnl_df["result"] = np.where(all_non_nan_constant_temperature_lt_0, 0, ann_constant_decision)

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const}min")
        fnl_df["result"] = fnl_df[fnl_df.index >= guard_datetime]["result"]

        conditions: pd.Series = all_non_nan_constant_temperature_lt_0 | all_non_nan_constant_temp_gt_0_hum_lt_85
        fnl_df["result"] = fnl_df.loc[conditions, "result"]

        fnl_df["ann_constant"] = (
            fnl_df["result"]
            .iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=0, closed="left")
            .apply(ConstantDataCheck.assign_first_value_in_window)
        )

        ann_constant_decision = np.where(all_non_nan_constant_temp_gt_0_hum_lt_85, 0, fnl_df["ann_constant_frozen"])
        fnl_df["result"] = np.where(all_non_nan_constant_temperature_lt_0, ann_constant_frozen, ann_constant_decision)

        fnl_df["result"] = fnl_df[fnl_df.index >= guard_datetime]["result"]
        fnl_df["result"] = fnl_df.loc[conditions, "result"]

        fnl_df["ann_constant_frozen"] = (
            fnl_df["result"]
            .iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=0, closed="left")
            .apply(ConstantDataCheck.assign_first_value_in_window)
        )

        fnl_df = fnl_df.drop(
            ["median_temperature", "median_humidity", "non_nan_count", "unique_values", "result"], axis=1
        )

        return fnl_df

    @staticmethod
    def check_constant_wind_speed(
        fnl_df: pd.DataFrame, time_window_const: int, ann_constant: int, ann_constant_frozen: int
    ) -> pd.DataFrame:
        """Check for constant values in a small time window for wind speed data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation that will be used for values identified as constant
            ann_constant_frozen (int): The annotation to be used for values identified as constant
                                        because of a frozen sensor

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which columns for constant data annotation are added
        """
        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const
        )

        all_non_nan_constant: pd.Series
        fnl_df, all_non_nan_constant = ConstantDataCheck.prepare_wind_df_and_condition(fnl_df, time_window_const)
        fnl_df.index = pd.to_datetime(fnl_df.index)

        temperature_lt_0: pd.Series = fnl_df["median_temperature"] <= 0
        wind_speed_all_0: pd.Series = (
            fnl_df["wind_speed_for_raw_check"]
            .eq(0)
            .rolling(window=f"{time_window_const}min", min_periods=1, closed="left")
            .sum()
            == time_window_const_as_row_count
        )
        temperature_lt_0_wind_speed_all_0: pd.Series = temperature_lt_0 & wind_speed_all_0

        temp_gt_0_hum_lt_85_wind_speed_all_0: pd.Series = (
            (fnl_df["median_temperature"] > 0) & (fnl_df["median_humidity"] < 85) & wind_speed_all_0  # noqa: PLR2004
        )

        wind_speed_all_not_0: pd.Series = (
            fnl_df["wind_speed_for_raw_check"]
            .ne(0)
            .rolling(window=f"{time_window_const}min", min_periods=1, closed="left")
            .sum()
            == time_window_const_as_row_count
        )

        all_non_nan_constant_wind_speed_all_not_0: pd.Series = all_non_nan_constant & wind_speed_all_not_0
        all_non_nan_constant_temp_gt_0_hum_lt_85_wind_speed_all_0: pd.Series = (
            all_non_nan_constant & temp_gt_0_hum_lt_85_wind_speed_all_0
        )
        all_non_nan_constant_temperature_lt_0_wind_speed_all_0: pd.Series = (
            all_non_nan_constant & temperature_lt_0_wind_speed_all_0
        )

        # Highest priority in an if-else is used last
        ann_constant_decision: np.ndarray = np.where(
            all_non_nan_constant_wind_speed_all_not_0, ann_constant, fnl_df["ann_constant"]
        )
        ann_constant_decision = np.where(
            all_non_nan_constant_temp_gt_0_hum_lt_85_wind_speed_all_0, ann_constant, ann_constant_decision
        )

        fnl_df["result"] = np.where(all_non_nan_constant_temperature_lt_0_wind_speed_all_0, 0, ann_constant_decision)

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const}min")
        fnl_df["result"] = fnl_df[fnl_df.index >= guard_datetime]["result"]

        conditions: pd.Series = (
            all_non_nan_constant_temperature_lt_0_wind_speed_all_0
            | all_non_nan_constant_temp_gt_0_hum_lt_85_wind_speed_all_0
            | all_non_nan_constant_wind_speed_all_not_0
        )
        fnl_df["result"] = fnl_df.loc[conditions, "result"]

        fnl_df["ann_constant"] = (
            fnl_df["result"]
            .iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=0, closed="left")
            .apply(ConstantDataCheck.assign_first_value_in_window)
        )

        ann_constant_decision = np.where(all_non_nan_constant_wind_speed_all_not_0, 0, fnl_df["ann_constant_frozen"])
        ann_constant_decision = np.where(
            all_non_nan_constant_temp_gt_0_hum_lt_85_wind_speed_all_0, 0, ann_constant_decision
        )

        fnl_df["result"] = np.where(
            all_non_nan_constant_temperature_lt_0_wind_speed_all_0, ann_constant_frozen, ann_constant_decision
        )

        fnl_df["result"] = fnl_df[fnl_df.index >= guard_datetime]["result"]

        conditions = (
            all_non_nan_constant_temperature_lt_0_wind_speed_all_0
            | all_non_nan_constant_temp_gt_0_hum_lt_85_wind_speed_all_0
            | all_non_nan_constant_wind_speed_all_not_0
        )
        fnl_df["result"] = fnl_df.loc[conditions, "result"]

        fnl_df["ann_constant_frozen"] = (
            fnl_df["result"]
            .iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=0, closed="left")
            .apply(ConstantDataCheck.assign_first_value_in_window)
        )

        fnl_df = fnl_df.drop(
            ["median_temperature", "median_humidity", "non_nan_count", "unique_values", "result"], axis=1
        )

        return fnl_df

    @staticmethod
    def check_constant_illuminance(fnl_df: pd.DataFrame, time_window_const: int, ann_constant: int) -> pd.DataFrame:
        """Check for constant values in a small time window for illuminance data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation that will be used for values identified as constant

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which a column for constant data annotation is added
        """
        fnl_df.index = pd.to_datetime(fnl_df.index)

        # Create a column for the non-zero values in the rolling window
        fnl_df["non_zero"] = (
            fnl_df["illuminance_for_raw_check"]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(lambda x: (x != 0).all())
        )

        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const
        )

        # Filter rows based on conditions
        condition: pd.Series = (
            (fnl_df["non_nan_count"] == time_window_const_as_row_count)  # all non-nan
            & (fnl_df["unique_values"] == 1)  # all values constant
            & fnl_df["non_zero"]  # all values non-zero
        )

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const}min")

        annotation_condition_reversed: pd.Series = (
            condition.iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(lambda x: x[x.index >= guard_datetime].sum())
        )
        fnl_df["ann_constant"] = (ann_constant * annotation_condition_reversed > 0).astype(int)[::-1]

        fnl_df = fnl_df.drop(["non_nan_count", "unique_values", "non_zero"], axis=1)

        return fnl_df

    @staticmethod
    def check_constant_miscellaneous(fnl_df: pd.DataFrame, time_window_const: int, ann_constant: int) -> pd.DataFrame:
        """Check for constant values in a small time-window for other variables' data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation that will be used for values identified as constant

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which a column for constant data annotation is added
        """
        fnl_df.index = pd.to_datetime(fnl_df.index)

        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const
        )

        # Filter rows based on conditions all values non-nan and all values constant
        condition: pd.Series[bool] = (fnl_df["non_nan_count"] == time_window_const_as_row_count) & (
            fnl_df["unique_values"] == 1
        )

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const}min")

        annotation_condition_reversed: pd.Series = (
            condition.iloc[::-1]
            .rolling(f"{time_window_const}min", min_periods=1, closed="left")
            .apply(lambda x: x[x.index >= guard_datetime].sum())
        )
        fnl_df["ann_constant"] = (ann_constant * annotation_condition_reversed > 0).astype(int)[::-1]

        fnl_df = fnl_df.drop(["non_nan_count", "unique_values"], axis=1)

        return fnl_df

    @staticmethod
    def check_constant_temperature_day(
        fnl_df: pd.DataFrame, time_window_const_max: int, ann_constant_max: int
    ) -> pd.DataFrame:
        """Check for constant values in a daily time window for temperature data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            time_window_const_max (int): The time window to search for constant data [in minutes]
            ann_constant_max (int): The annotation that will be used for values identified as constant

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which a column for constant data annotation is added
        """
        fnl_df.index = pd.to_datetime(fnl_df.index)

        # Create a column for the non-NaN count in the rolling window
        fnl_df["non_nan_count"] = (
            fnl_df["temperature_for_raw_check"]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(lambda x: x.count())
        )

        # Create a column for the unique values in the rolling window
        fnl_df["unique_values"] = (
            fnl_df["temperature_for_raw_check"]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(lambda x: x.nunique())
        )

        # Count the number of rows for a day of the dataframe using as reference the last timestamp of the df
        time_window_const_max_as_row_count: int = ConstantDataCheck.get_number_of_rows_of_last_day(
            fnl_df, time_window_const_max
        )

        # Filter rows based on conditions
        condition: pd.Series[bool] = (fnl_df["non_nan_count"] == time_window_const_max_as_row_count) & (
            fnl_df["unique_values"] == 1
        )

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const_max}min")

        annotation_condition_reversed: pd.Series = (
            condition.iloc[::-1]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(lambda x: x[x.index >= guard_datetime].sum())
        )

        fnl_df["ann_constant_long"] = (ann_constant_max * annotation_condition_reversed > 0).astype(int)[::-1]

        fnl_df = fnl_df.drop(["non_nan_count", "unique_values"], axis=1)

        return fnl_df

    @staticmethod
    def check_constant_wind_day(
        fnl_df: pd.DataFrame, parameter: str, time_window_const_max: int, ann_constant_max: int
    ) -> pd.DataFrame:
        """Check for constant values in a daily time window for wind data.

        Args:
        ----
            fnl_df (pd.DataFrame): The dataframe containing the data.
            parameter (str): The name of the examined parameter
            time_window_const_max (int): The time window to search for constant data [in minutes]
            ann_constant_max (int): The annotation that will be used for values identified as constant

        Returns:
        -------
            pd.DataFrame: The original dataframe, to which a column for constant data annotation is added
        """
        fnl_df.index = pd.to_datetime(fnl_df.index)

        fnl_df["unique_values"] = (
            fnl_df[f"{parameter}_for_raw_check"]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(lambda x: x.nunique())
        )

        fnl_df["median"] = (
            fnl_df["temperature_for_raw_check"]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(np.nanmedian)
        )

        # Filter rows based on conditions
        condition: pd.Series[bool] = (fnl_df["unique_values"] == 1) & (fnl_df["median"] > 0)

        guard_datetime: pd.Timestamp = fnl_df.index[0] + pd.Timedelta(f"{time_window_const_max}min")

        annotation_condition_reversed: pd.Series = (
            condition.iloc[::-1]
            .rolling(f"{time_window_const_max}min", min_periods=1, closed="left")
            .apply(lambda x: x[x.index >= guard_datetime].sum())
        )

        fnl_df["ann_constant_long"] = (ann_constant_max * annotation_condition_reversed > 0).astype(int)[::-1]

        # Reset index
        fnl_df = fnl_df.drop(["unique_values", "median"], axis=1)

        return fnl_df

    @staticmethod
    def constant_data_check(
        fnl_df: pd.DataFrame,
        parameter: str,
        time_window_constant: int,
        ann_constant: int,
        ann_constant_frozen: int,
        rh_threshold: float,
        time_window_constant_max: int,
        ann_constant_max: int,
    ) -> pd.DataFrame:
        """Detects constant values within a certain time window.

        If the dataframe used is the one produced by filling_ignoring_period(), then nan
        for certain time windows may be ignored, as they have been filled with actual values.
        When no data are found, a new search for constant data further in the rolling time
        window is performed.

        Args:
        ----
            fnl_df (pd.DataFrame): a dataframe with a fixed temporal resolution, containing the data
            parameter (str): the name of the examined parameter
            time_window_constant (int): The time window to search for constant data [in minutes]
            ann_constant (int): The annotation to be used where data are constant for the time_window_constant
                                period of time
            ann_constant_frozen (int): The annotation to be used where wind data are constant for the
                                        time_window_constant certain period of time when temperature<0C
            rh_threshold (float): the threshold to exclude annotating data as constant, when e.g., RH>95% [in %]
            time_window_constant_max (int): The bigger time window (e.g. a whole day) to search for constant
                                            data [in minutes]
            ann_constant_max (int): The annotation to be used where data are constant for the
                                    time_window_constant_max period of time

        Returns:
        -------
            pd.DataFrame: a df with all the parameters, but with two columns for annotating constant values
                            and constant values under freezing conditions (the latter is always 0 unless the
                            parameter is wind speed/direction) only for the selected parameter
        """
        # Now we annotate with "ann_constant" or "ann_constant_frozen" the constant data within
        # the "time_window_constant" period
        # We consider that if the gap is longer than the "time_window_constant", we cannot guess the values
        # and the rolling window starts searching for a new "time_window_constant" period of constant values

        fnl_df["ann_constant"] = 0  # This is for checking constant values for short period
        fnl_df["ann_constant_long"] = 0  # This is for checking constant values for long period
        fnl_df["ann_constant_frozen"] = 0  # This is for checking constant values under frozen conditions

        fnl_df = fnl_df.set_index("date").sort_index()

        # Create a column for the non-NaN count in the rolling window
        fnl_df["non_nan_count"] = (
            fnl_df[f"{parameter}_for_raw_check"]
            .rolling(f"{time_window_constant}min", min_periods=1, closed="left")
            .apply(lambda x: x.count())
        )

        # Create a column for the unique values in the rolling window
        fnl_df["unique_values"] = (
            fnl_df[f"{parameter}_for_raw_check"]
            .rolling(f"{time_window_constant}min", min_periods=1, closed="left")
            .apply(lambda x: x.nunique())
        )

        if parameter == "humidity":
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_humidity_temperature(
                fnl_df, time_window_constant, ann_constant, rh_threshold
            )

        elif parameter == "temperature":
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_humidity_temperature(
                fnl_df, time_window_constant, ann_constant, rh_threshold
            )

            # perform the check within the big rolling time window (day)
            fnl_df = ConstantDataCheck.check_constant_temperature_day(
                fnl_df, time_window_constant_max, ann_constant_max
            )

        elif parameter == "wind_direction":
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_wind_direction(
                fnl_df, time_window_constant, ann_constant, ann_constant_frozen
            )

            # perform the check within the big rolling time window (day)
            fnl_df = ConstantDataCheck.check_constant_wind_day(
                fnl_df, parameter, time_window_constant_max, ann_constant_max
            )

        elif parameter == "wind_speed":
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_wind_speed(
                fnl_df, time_window_constant, ann_constant, ann_constant_frozen
            )

            # perform the check within the big rolling time window (day)
            fnl_df = ConstantDataCheck.check_constant_wind_day(
                fnl_df, parameter, time_window_constant_max, ann_constant_max
            )

        elif parameter == "illuminance":
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_illuminance(fnl_df, time_window_constant, ann_constant)

        else:
            # perform the check within the small rolling time window
            fnl_df = ConstantDataCheck.check_constant_miscellaneous(fnl_df, time_window_constant, ann_constant)

        fnl_df = fnl_df.reset_index(drop=False, names=["date"])

        return fnl_df
