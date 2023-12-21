from __future__ import annotations

import numpy as np
import pandas as pd
import numpy.typing as npt

from obc_sqc.model.averaging_utils import AveragingUtils


class HourAveraging:
    """Functions for calculating the hour averaging from minute averaging data."""

    @staticmethod
    def calculate_wind_u(wind_speed_avg: pd.Series, wind_direction_avg: pd.Series) -> pd.Series:
        """Calculate the u component of wind based on wind speed and wind direction.

        Args:
        ----
            wind_speed_avg (pd.Series): A NumPy array or pandas Series containing wind speed values.
            wind_direction_avg (pd.Series): A NumPy array or pandas Series containing wind direction values.

        Returns:
        -------
            pd.Series: A Series representing the calculated u component of wind.
        """
        wind_u: pd.Series = -1 * wind_speed_avg * np.sin(wind_direction_avg.astype("Float64") * np.pi / 180.0)
        return wind_u

    @staticmethod
    def calculate_wind_v(wind_speed_avg: pd.Series, wind_direction_avg: pd.Series) -> pd.Series:
        """Calculate the v component of wind based on wind speed and wind direction.

        Args:
        ----
            wind_speed_avg (pd.Series): A NumPy array or pandas Series containing wind speed values.
            wind_direction_avg (pd.Series): A NumPy array or pandas Series containing wind direction values.

        Returns:
        -------
            pd.Series: A Series representing the calculated v component of wind.
        """
        wind_v: pd.Series = -1 * wind_speed_avg * np.cos(wind_direction_avg.astype("Float64") * np.pi / 180.0)
        return wind_v

    @staticmethod
    def wind_averaging(
        minute_averaging: pd.DataFrame, fnl_timeslot: int, availability_threshold: float, delim: str
    ) -> pd.DataFrame:
        """Calculate wind hour averages.

        Args:
        ----
            minute_averaging (pd.DataFrame): The DataFrame containing minute averaged data
            fnl_timeslot (int): the length of the final timeframe resolution in [minutes],
                                    e.g., if we want to give rewards for each hour then the
                                    value is 60 minutes
            availability_threshold (float): the availability threshold, e.g., if <67% of
                                            timeslots within a certain period is available,
                                            averaging or rewarding is not possible [x out of 1]
            delim (str): The delimeter used between annotations.

        Returns:
        -------
            pd.DataFrame: The hour averaged DataFrame
        """
        # calculate the u and v components of the wind in the raw dataset
        minute_averaging["wind_u"] = HourAveraging.calculate_wind_u(
            minute_averaging["wind_speed_avg"], minute_averaging["wind_direction_avg"]
        )

        minute_averaging["wind_v"] = HourAveraging.calculate_wind_v(
            minute_averaging["wind_speed_avg"], minute_averaging["wind_direction_avg"]
        )

        # Group by x-minutes and calculate average of wind components
        # It is important that we firstly average the u and v components of the wind
        hour_averaging: pd.DataFrame = minute_averaging.groupby(
            pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min")
        ).agg(
            u=("wind_u", "mean"),  # averages the u component of the wind
            v=("wind_v", "mean"),  # averages the v component of the wind
            num_time_slots=(
                "utc_datetime",
                "count",
            ),  # counts the elements having a value
            num_hourly_faulty=(
                "ann_total",
                "sum",
            ),  # finds the total faulty elements within an hour
            num_hourly_faulty_rewards=(
                "ann_total_rewards",
                "sum",
            ),  # finds the total reward-faulty elements within an hour
            annotation=(
                "annotation",
                lambda x: delim.join(s for s in list(x.str.split(delim).sum()) if s),
                # merges all text annotations in one string
            ),
        )

        # Both wind speed and wind direction average columns will be needed in
        # the hour averaging of both wind speed and direction.
        hour_averaging.insert(
            0,
            "wind_speed_avg",
            hour_averaging.apply(AveragingUtils.row_wind_speed_calculation, axis=1),
        )
        hour_averaging.insert(
            0,
            "wind_direction_avg",
            hour_averaging.apply(AveragingUtils.row_wind_direction_calculation, axis=1),
        )

        hour_averaging["u"] = hour_averaging["u"].round(2)
        hour_averaging["v"] = hour_averaging["v"].round(2)
        hour_averaging["wind_speed_avg"] = hour_averaging["wind_speed_avg"].astype("Float64").round(2)
        hour_averaging["wind_direction_avg"] = hour_averaging["wind_direction_avg"].astype("Float64").round(2)

        # Calulate corrected wind speed and wind direction average, using only non-faulty
        # data based on the annotations
        hour_averaging["wind_spd_avg_corrected"] = minute_averaging.groupby(
            pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min")
        ).apply(
            lambda x: AveragingUtils.column_wind_speed_average_using_annotation(
                x, availability_threshold=availability_threshold, annotation_col="ann_total"
            )
        )

        hour_averaging["wind_dir_avg_corrected"] = minute_averaging.groupby(
            pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min")
        ).apply(
            lambda x: AveragingUtils.column_wind_direction_average_using_annotation(
                x, availability_threshold=availability_threshold, annotation_col="ann_total"
            )
        )

        return hour_averaging

    @staticmethod
    def precipitation_averaging(
        minute_averaging: pd.DataFrame, fnl_timeslot: int, parameter: str, delim: str
    ) -> pd.DataFrame:
        """Calculate precipitation hour average.

        Args:
        ----
            minute_averaging (pd.DataFrame): The DataFrame containing minute averaged data
            fnl_timeslot (int): the length of the final timeframe resolution in [minutes],
                                    e.g., if we want to give rewards for each hour then the
                                    value is 60 minutes
            parameter (str): the name of the parameter
            delim (str): The delimeter used between annotations.

        Returns:
        -------
            pd.DataFrame: The hour averaged DataFrame
        """
        # Group by x-minutes and calculate metadata for accumulated precipitation
        hour_averaging: pd.DataFrame = minute_averaging.groupby(
            pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min")
        ).agg(
            precipitation_accumulated_avg=(f"{parameter}_avg", "sum"),
            num_time_slots=(
                "utc_datetime",
                "count",
            ),  # counts the elements within the selected period e.g., 60minutes
            num_hourly_faulty=(
                "ann_total",
                "sum",
            ),  # finds the total faulty elements within an hour
            num_hourly_faulty_rewards=(
                "ann_total_rewards",
                "sum",
            ),  # finds the total reward-faulty elements within an hour
            annotation=(
                "annotation",
                lambda x: delim.join(s for s in list(x.str.split(delim).sum()) if s),
                # merges all text annotations in one string
            ),
        )

        return hour_averaging

    @staticmethod
    def averaging(
        minute_averaging: pd.DataFrame,
        fnl_timeslot: int,
        availability_threshold: float,
        parameter: str,
        delim: str,
    ) -> pd.DataFrame:
        """Calculate hour average.

        Args:
        ----
            minute_averaging (pd.DataFrame): The DataFrame containing minute averaged data
            fnl_timeslot (int): the length of the final timeframe resolution in [minutes],
                                    e.g., if we want to give rewards for each hour then the
                                    value is 60 minutes
            availability_threshold (float): the availability threshold, e.g., if <67% of
                                            timeslots within a certain period is available,
                                            averaging or rewarding is not possible [x out of 1]
            parameter (str): the name of the parameter
            delim (str): The delimeter used between annotations.

        Returns:
        -------
            pd.DataFrame: The hour averaged DataFrame
        """
        # Group by x-minutes and calculate metadata for other variables
        hour_averaging: pd.DataFrame = (
            minute_averaging.groupby(pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min"))
            .agg(
                parameter_avg_name=(f"{parameter}_avg", "mean"),
                # counts the simple average of the parameter
                num_time_slots=(
                    "utc_datetime",
                    "count",
                ),  # counts the elements within the selected period e.g., 60minutes
                num_hourly_faulty=(
                    "ann_total",
                    "sum",
                ),  # finds the total faulty elements within an hour
                num_hourly_faulty_rewards=(
                    "ann_total_rewards",
                    "sum",
                ),  # finds the total reward-faulty elements within an hour
                annotation=(
                    "annotation",
                    lambda x: delim.join(s for s in list(x.str.split(delim).sum()) if s),
                ),  # merges all text annotation in one string
            )
            .rename(columns={"parameter_avg_name": f"{parameter}_avg"})
        )

        hour_averaging[f"{parameter}_avg"] = hour_averaging[f"{parameter}_avg"].round(2)

        # Caluclate corrected accumulated precipitation, using only non-faulty
        # data based on the annotations
        hour_averaging[f"{parameter}_avg_corrected"] = minute_averaging.groupby(
            pd.Grouper(key="utc_datetime", freq=f"{fnl_timeslot}min")
        ).apply(
            lambda x: AveragingUtils.column_average_using_annotation(
                x,
                column=f"{parameter}_avg_corrected",
                availability_threshold=availability_threshold,
                annotation_col="ann_total",
            )
        )

        hour_averaging[f"{parameter}_avg_corrected"] = hour_averaging[f"{parameter}_avg_corrected"].round(2)

        return hour_averaging

    @staticmethod
    def calculate_valid_percentage(num_time_slots: pd.Series, num_hourly_faulty: pd.Series) -> pd.Series:
        """Calculate the percentage of valid observations per hour.

        Args:
        ----
            num_time_slots (pd.Series): The number of time slots per hour
            num_hourly_faulty (pd.Series): The number of faulty slots per hour

        Returns:
        -------
            pd.Series: The percentage of valid observations per hour
        """
        result: pd.Series = ((num_time_slots - num_hourly_faulty) * 100) / num_time_slots

        return result

    @staticmethod
    def calculate_ann_total_hour(
        num_time_slots: pd.Series, num_hourly_faulty: pd.Series, availability_threshold: float
    ) -> npt.NDArray:
        """Annotate an hourly timeslot with 1 if < availability_threshold of the data is available.

        Otherwise, annotate with 0.

        Args:
        ----
            num_time_slots (pd.Series): The number of time slots per hour
            num_hourly_faulty (pd.Series): The number of faulty slots per hour
            availability_threshold (float): The desired threshold

        Returns:
        -------
            numpy.typing.NDArray: The annotations
        """
        result: pd.Series = (num_time_slots - num_hourly_faulty) / num_time_slots
        annotations: npt.NDArray = np.where(result < availability_threshold, 1, 0)
        return annotations

    @staticmethod
    def calculate_valid_percentage_rewards(num_time_slots: pd.Series, num_hourly_faulty: pd.Series) -> pd.Series:
        """Calculate the percentage of valid observations per hour.

        Args:
        ----
            num_time_slots (pd.Series): The number of time slots per hour
            num_hourly_faulty (pd.Series): The number of faulty slots per hour

        Returns:
        -------
            pd.Series: The percentage of valid observations per hour
        """
        result: pd.Series = ((num_time_slots - num_hourly_faulty) * 100) / num_time_slots
        return result

    @staticmethod
    def calculate_ann_total_hour_rewards(
        num_time_slots: pd.Series, num_hourly_faulty: pd.Series, availability_threshold: float
    ) -> npt.NDArray:
        """Annotate an hourly timeslot with 1 if < availability_threshold of the data is available.

        Otherwise, annotate with 0.

        Args:
        ----
            num_time_slots (pd.Series): The number of time slots per hour
            num_hourly_faulty (pd.Series): The number of faulty slots per hour
            availability_threshold (float): The desired threshold

        Returns:
        -------
            numpy.typing.NDArray: The annotations
        """
        result: pd.Series = (num_time_slots - num_hourly_faulty) / num_time_slots
        annotations: npt.NDArray = np.where(result < availability_threshold, 1, 0)
        return annotations

    @staticmethod
    def hour_averaging(
        minute_averaging: pd.DataFrame, fnl_timeslot: int, availability_threshold: float, parameter: str
    ) -> pd.DataFrame:
        """Calculates the hourly averages of a parameter and the percentage of available data within an hour.

        Then, it annotates as invalid the hourly average if the percentage of available data was lower than the
        given availability_threshold. Note that the average of e.g. 01:00:00 is the average of values between
        00:00:01-01:00:00.

        Args:
        ----
            minute_averaging (pd.DataFrame): the output of minute_averaging function
            fnl_timeslot (int): the length of the final timeframe resolution in [minutes], e.g., if we want to
                                give rewards for each hour then the value is 60 minutes
            availability_threshold (float): the availability threshold, e.g., if <67% of timeslots within a certain
                                            period is available, averaging or rewarding is not possible [x out of 1]
            parameter (str): the parameter that this function looks into, e.g., temperature, humidity, wind speed etc.

        Returns:
        -------
            pd.DataFrame: a DataFrame containing:
                            a df with the a. averaged parameter under investigation,
                            b. number of total timeslots within an hour,
                            c. number of faulty slots within an hour,
                            d. the percentage of valid data,
                            e. numeric annotation (for both all- and -reward faulty data) and
                            f. text annotation of each hourly slot
        """
        delim: str = ","

        minute_averaging = minute_averaging.reset_index(names=["utc_datetime"])
        hour_averaging: pd.DataFrame

        # In case of wind, we need to apply vector average
        if parameter in {"wind_speed", "wind_direction"}:
            hour_averaging = HourAveraging.wind_averaging(minute_averaging, fnl_timeslot, availability_threshold, delim)

        # For the rest of parameters we calculate the simple average
        elif parameter == "precipitation_accumulated":
            hour_averaging = HourAveraging.precipitation_averaging(minute_averaging, fnl_timeslot, parameter, delim)
        else:
            hour_averaging = HourAveraging.averaging(
                minute_averaging, fnl_timeslot, availability_threshold, parameter, delim
            )

        hour_averaging["valid_percentage"] = HourAveraging.calculate_valid_percentage(
            hour_averaging["num_time_slots"], hour_averaging["num_hourly_faulty"]
        ).astype(float)

        hour_averaging["valid_percentage"] = hour_averaging["valid_percentage"].round(2)

        hour_averaging["ann_total_hour"] = HourAveraging.calculate_ann_total_hour(
            hour_averaging["num_time_slots"],
            hour_averaging["num_hourly_faulty"],
            availability_threshold,
        ).astype(int)

        hour_averaging["valid_percentage_rewards"] = HourAveraging.calculate_valid_percentage_rewards(
            hour_averaging["num_time_slots"], hour_averaging["num_hourly_faulty"]
        ).astype(float)

        hour_averaging["valid_percentage_rewards"] = hour_averaging["valid_percentage_rewards"].round(2)

        hour_averaging["ann_total_hour_rewards"] = HourAveraging.calculate_ann_total_hour_rewards(
            hour_averaging["num_time_slots"],
            hour_averaging["num_hourly_faulty"],
            availability_threshold,
        ).astype(int)

        # Rearranging location of some columns for improving the readability of csv
        column_to_move: str = "annotation"
        columns_ordered: list[str] = [col for col in hour_averaging.columns if col != column_to_move] + [column_to_move]
        hour_averaging = hour_averaging[columns_ordered]

        return hour_averaging
