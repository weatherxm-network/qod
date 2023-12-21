from __future__ import annotations

import numpy as np


class InitialParams:  # noqa: D101
    @staticmethod
    def picking_initial_parameters(station_type):
        """This def sets the appropriate parameterization for a given weather station model

        station_type (str): the weather station model (M5 or Helium)

        result: a series of variables and lists that correspond to a certain weather station model.
            See the description of each parameter in the next lines after the #"""  # noqa: D202, D208, D209, D400, D415

        ann_no_median = 2  # Annotation where no median has been calculated
        ann_no_datum = 3  # Annotation where no datum is available
        ann_invalid_datum = 4  # Annotation where the datum exists but it's not valid
        ann_constant = 5  # Annotation where data are constant for a certain period of time

        # Annotation where data are constant for the max possible period of time
        ann_constant_max = 7

        # Annotation where wind data are constant for a certain period of time when temperature<0 C
        ann_constant_frozen = 6

        # This is the length of the final timeframe resolution in [minutes], e.g., if we want to give
        # rewards for each hour then the value is 60 minutes
        fnl_timeslot = 60

        if station_type == "WS1000":
            data_timestep = 16  # This is the timestep of the raw data [in seconds]

            # This is the tolerance with which a raw datum corresponds to a standard new timeframe
            # with fixed timestep [seconds]
            time_tolerance = data_timestep // 2

            time_window_median = 10  # This is the time window for rolling median [in minutes]

            # This is the period within nans can be replaced with the latest valid value.
            # This is used in constant check [seconds]
            ignoring_period = 60
            start_timestamp = "18:00:00"

            # This is the threshold used in the def "check_for_constant_data"
            # to exclude annotating data as constant when e.g., RH>95%
            rh_threshold = 95
            pr_int = 0.254  # This is the rain gauge resolution [mm]

            # the time window between start_timestamp and the first timestamp of the current day [in minutes]
            preprocess_timewindow = 360

            parameters_for_testing = [
                "humidity",
                "temperature",
                "wind_direction",
                "wind_speed",
                "pressure",
                "illuminance",
                "precipitation_accumulated",
            ]

            # This is the availability threshold [x out of 1], e.g., if <66% of timeslots within
            # a certain period is available averaging or rewarding is not possible
            availability_threshold_median = [0.67, 0.67, 0.75, 0.75, 0.67, 0.67, np.nan]
            availability_threshold_m = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            availability_threshold_h = [0.67, 0.67, 0.75, 0.75, 0.67, 0.67, 0.85]

            # pr_hourly_availability = 0.85  # This is the availability threshold for precipitation in the hourly level

            # jump thresholds for checking raw data
            raw_control_thresholds = [5, 2, np.nan, 20, 0.3, 97600, np.nan]

            # jump thresholds for checking averaged data
            minute_control_thresholds = [10, 3, np.nan, 10, 0.5, 97600, np.nan]

            # the desired period for averaging each of the parameters
            minute_averaging_period = [1, 1, 2, 2, 1, 1, 1]

            # This is the time window for rolling window checking constant values within it [in minutes]
            time_window_constant = [
                360,
                240,
                360,
                360,
                120,
                120,
                np.nan,
            ]

            # This is the maximum time window for rolling window checking constant values within it [in minutes]
            time_window_constant_max = [
                np.nan,
                1440,
                1440,
                1440,
                1440,
                np.nan,
                np.nan,
            ]

            obc_limits = [
                [10, -40, 0, 0, 300, 0, 0],
                [99, 60, 359, 50, 1100, 400000, pr_int * data_timestep],
            ]

        elif station_type == "WS2000":
            # This is the timestep of the raw data [in seconds]
            data_timestep = 180

            # 3 #This is the tolerance with which a raw datum corresponds to a
            # standard new timeframe with fixed timestep [seconds]
            time_tolerance = data_timestep // 2

            # This is the time window for rolling median [in minutes]
            time_window_median = 60
            start_timestamp = "18:00:00"

            # This is the period within nans can be replaced with the latest valid value.
            # This is used in constant check [seconds]
            ignoring_period = 180

            # This is the threshold used in the def "check_for_constant_data"
            # to exclude annotating data as constant when e.g., RH>95%
            rh_threshold = 95

            # This is the rain gauge resolution [mm]
            pr_int = 0.254

            # the time window between start_timestamp and the first timestamp of the current day [in minutes]
            preprocess_timewindow = 360

            parameters_for_testing: list[str] = [
                "humidity",
                "temperature",
                "wind_direction",
                "wind_speed",
                "pressure",
                "illuminance",
                "precipitation_accumulated",
            ]

            # This is the availability threshold [x out of 1], e.g., if <66% of timeslots
            # within a certain period is available averaging or rewarding is not possible
            availability_threshold_median = [0.67, 0.67, 0.75, 0.75, 0.67, 0.67, np.nan]
            availability_threshold_m = [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ]  # this is not used in WS2000, it's given only for not breaking the processes
            availability_threshold_h = [
                0.67,
                0.67,
                0.75,
                0.75,
                0.67,
                0.67,
                0.85,
            ]

            # pr_hourly_availability = 0.85  # This is the availability threshold for precipitation in the hourly level

            # jump thresholds for checking raw data
            raw_control_thresholds = [10, 3, 10, 10, 0.5, 97600, np.nan]

            # jump thresholds for checking averaged data
            minute_control_thresholds = [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]

            # the desired period for averaging each of the parameters
            minute_averaging_period = [
                60,
                60,
                60,
                60,
                60,
                60,
                60,
            ]

            # This is the time window for rolling window checking constant values within it [in minutes]
            time_window_constant = [
                360,
                240,
                360,
                360,
                120,
                120,
                np.nan,
            ]

            # This is the maximum time window for rolling window checking constant values within it [in minutes]
            time_window_constant_max = [
                np.nan,
                1440,
                1440,
                1440,
                1440,
                np.nan,
                np.nan,
            ]

            obc_limits = [
                [1, -40, 0, 0, 540, 0, 0],
                [99, 80, 359, 50, 1100, 200000, pr_int * data_timestep],
            ]

            # upper thresholds that should not be exceeded e.g., per hour for each of the parameters
            upper_thresholds = [
                80,
                15,
                np.nan,
                15,
                15,
                146400,
                np.nan,
            ]

            # period that the raw_control_thresholds (which here are the minute thresholds)
            # are initially applied to [minutes]
            initial_threshold_period = [
                1,
                1,
                2,
                2,
                1,
                1,
                1,
            ]

            # We reduce the thresholds depending on the data time resolution
            raw_control_thresholds = [
                min(
                    upper_thresholds[i],
                    (raw_control_thresholds[i] * (data_timestep / 60) / initial_threshold_period[i]),
                )
                for i in range(len(raw_control_thresholds))
            ]

        return (
            ann_no_median,
            ann_no_datum,
            ann_invalid_datum,
            ann_constant,
            ann_constant_frozen,
            data_timestep,
            time_tolerance,
            time_window_median,
            ignoring_period,
            fnl_timeslot,
            rh_threshold,
            parameters_for_testing,
            availability_threshold_median,
            availability_threshold_m,
            availability_threshold_h,
            raw_control_thresholds,
            minute_control_thresholds,
            minute_averaging_period,
            time_window_constant,
            obc_limits,
            start_timestamp,
            time_window_constant_max,
            ann_constant_max,
            pr_int,
            preprocess_timewindow,
        )
