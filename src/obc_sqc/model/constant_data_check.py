from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantDataCheck:  # noqa: D101
    @staticmethod
    def check_for_constant_data(  # noqa: C901, PLR0912, PLR0915
        fnl_df,
        parameter,
        data_timestep,
        time_window_constant,
        ann_constant,
        ann_constant_frozen,
        rh_threshold,
        time_window_constant_max,
        ann_constant_max,
    ):
        """This def detects constant values within a certain time window ignoring
        if there is no data in a single timeslot. However, for two or more consecutive
        no-data timeslots, the rolling window starts searching for a new 2-hour period
        of constant values

        fnl_df (df): the output of time_normalisation_dataframe def, which is a dataframe with a fixed temporal
            resolution
        parameter (str): the parameter that this def looks into, e.g., temperature, humidity, wind speed etc.
        data_timestep (int): the desired timestep of the final df [in seconds]
        time_window_constant (int): the time window for rolling window checking constant values within it [in minutes]
        ann_constant (int): annotation where data are constant for a certain period of time
        ann_constant_frozen (int): annotation where wind data are constant for a certain period of time when
            temperature<0C
        rh_threshold (int): the threshold to exclude annotating data as constant, when e.g., RH>95% [in %]

        result (df): a df with all the parameters, but with two columns for annotating constant values
                    and constant values under freezing conditions (the latter is always 0 unless the parameter
                    is wind spd/dir) only for the selected parameter"""  # noqa: D202, D205, D208, D209, D400, D415

        # Now we annotate with "ann_constant" or "ann_constant_frozen" the constant data within
        # the "time_window_constant" period
        # We consider that if the gap is longer than the "time_window_constant", we cannot guess the values
        # and the rolling window starts searching for a new "time_window_constant" period of constant values

        fnl_df.loc[:, "ann_constant"] = 0  # This is for checking constant values for short period
        fnl_df.loc[:, "ann_constant_long"] = 0  # This is for checking constant values for long period
        fnl_df.loc[:, "ann_constant_frozen"] = 0  # This is for checking constant values under frozen conditions

        # Count the number of rows between a day of dataframe using as reference the last timestamp of the df
        if not np.isnan(time_window_constant):
            last_timestamp = fnl_df["date"].iloc[-1]
            start_timestamp = last_timestamp - pd.Timedelta(
                minutes=time_window_constant
            )  # FIXME Last timestamp is 23:59:44 (not midnight)
            filtered_df = fnl_df[(fnl_df["date"] >= start_timestamp) & (fnl_df["date"] <= last_timestamp)]
            time_window_const_as_row_count = len(filtered_df)
        else:
            time_window_const_as_row_count = np.nan

        if not np.isnan(time_window_constant_max):
            last_timestamp = fnl_df["date"].iloc[-1]
            start_timestamp = last_timestamp - pd.Timedelta(minutes=time_window_constant_max)
            filtered_df = fnl_df[(fnl_df["date"] >= start_timestamp) & (fnl_df["date"] <= last_timestamp)]
            time_window_const_max_as_row_count = len(filtered_df)
        else:
            time_window_const_max_as_row_count = np.nan

            # apply rolling window and check if all elements in the window are the same
        for i in range(len(fnl_df)):
            if i < time_window_const_as_row_count:
                continue
            else:
                window_start = i - time_window_const_as_row_count
                window_end = i
                window_data = fnl_df.loc[window_start:window_end, f"{parameter}_for_raw_check"]

                # Our analysis suggests that when RH>=85%, then RH can be constant for hours,
                # otherwise it should be considered as faulty
                # If there is a series of constant values e.g., [94, 94, 94 .... 94, 85, 85, 85, 84],
                # All 94 values are annotated as constant and all 85 values as faulty, as the 94-85>5.
                # The first valid value will be the 84%.
                if parameter == "humidity":
                    if (
                        window_data.isna().sum() == 0
                        and len(np.unique(window_data[~np.isnan(window_data)])) == 1
                        and np.nanmedian(window_data) < rh_threshold
                    ):
                        fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant

                elif parameter == "temperature":
                    window_data_humidity = fnl_df.loc[window_start:window_end, "humidity_for_raw_check"]

                    if (
                        window_data.isna().sum() == 0
                        and len(np.unique(window_data[~np.isnan(window_data)])) == 1
                        and np.nanmedian(window_data_humidity) < rh_threshold
                    ):
                        fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant

                elif parameter == "temperature":
                    window_data_humidity = fnl_df.loc[window_start:window_end, "humidity_for_raw_check"]

                    if (
                        window_data.isna().sum() == 0
                        and len(np.unique(window_data)) == 1
                        and np.median(window_data_humidity) < rh_threshold
                    ):
                        fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant

                # For wind speed and direction we check if the data were constant under freezing or not
                # conditions and annotate with "ann_constant_frozen" and "ann_constant" respectively
                elif parameter == "wind_direction":
                    window_data_temperature = fnl_df.loc[window_start:window_end, "temperature_for_raw_check"]
                    window_data_humidity = fnl_df.loc[window_start:window_end, "humidity_for_raw_check"]

                    if window_data.isna().sum() == 0 and len(np.unique(window_data[~np.isnan(window_data)])) == 1:
                        if np.nanmedian(window_data_temperature) <= 0:  # if np.all(window_data_temperature <0 ):
                            fnl_df.loc[window_start:window_end, "ann_constant_frozen"] = ann_constant_frozen
                            fnl_df.loc[window_start:window_end, "ann_constant"] = 0
                        elif (
                            np.nanmedian(window_data_temperature) > 0
                            and np.nanmedian(window_data_humidity) < 85  # noqa: PLR2004
                        ):
                            fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant
                            fnl_df.loc[window_start:window_end, "ann_constant_frozen"] = 0

                elif parameter == "wind_speed":
                    window_data_temperature = fnl_df.loc[window_start:window_end, "temperature_for_raw_check"]
                    window_data_humidity = fnl_df.loc[window_start:window_end, "humidity_for_raw_check"]

                    if window_data.isna().sum() == 0 and len(np.unique(window_data[~np.isnan(window_data)])) == 1:
                        if (
                            np.nanmedian(window_data_temperature) <= 0 and (window_data == 0).all()
                        ):  # this is for frozen wind gauge
                            fnl_df.loc[window_start:window_end, "ann_constant_frozen"] = ann_constant_frozen
                            fnl_df.loc[window_start:window_end, "ann_constant"] = 0
                            if (
                                np.nanmedian(window_data_humidity) >= 85  # noqa: PLR2004
                            ):  # because it could not be forzen but it would be due to bad deployment
                                fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant
                        elif (
                            np.nanmedian(window_data_temperature) > 0
                            and np.nanmedian(window_data_humidity) < 85  # noqa: PLR2004
                            and (window_data == 0).all()
                        ):  # this may be due to a bad deployment
                            fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant
                            fnl_df.loc[window_start:window_end, "ann_constant_frozen"] = 0
                        elif (
                            window_data != 0
                        ).all():  # if wind_spd is constantly equal to a value =!0. This is a faulty sensor
                            fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant
                            fnl_df.loc[window_start:window_end, "ann_constant_frozen"] = 0

                elif parameter == "illuminance":
                    if (
                        window_data.isna().sum() == 0
                        and len(np.unique(window_data[~np.isnan(window_data)])) == 1
                        and (window_data != 0).all()
                    ):
                        fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant
                # For all the other parameters, we apply a simple check for constant values
                else:  # noqa: PLR5501
                    if window_data.isna().sum() == 0 and len(np.unique(window_data[~np.isnan(window_data)])) == 1:
                        fnl_df.loc[window_start:window_end, "ann_constant"] = ann_constant

        # Checking within the max time window that parameters could be constant.
        # For wind speed, wind direction and illuminance all the scenario are covered in the previous lines
        if parameter == "temperature" or parameter == "wind_speed" or parameter == "wind_direction":  # noqa: PLR1714
            for i in range(len(fnl_df)):
                if i < time_window_const_max_as_row_count:
                    continue
                else:
                    window_start = i - time_window_const_max_as_row_count
                    window_end = i
                    window_data = fnl_df.loc[window_start:window_end, f"{parameter}_for_raw_check"]

                    if parameter == "wind_speed" or parameter == "wind_direction":  # noqa: PLR1714
                        window_data_temperature = fnl_df.loc[window_start:window_end, "temperature_for_raw_check"]

                        # if wspd is constant for a whole day, it's suspicious. If it's frozen, it's annotated earlier
                        nan_median = np.median(window_data_temperature[~window_data_temperature.isna()])
                        if nan_median > 0 and len(np.unique(window_data[~np.isnan(window_data)])) == 1:
                            fnl_df.loc[window_start:window_end, "ann_constant_long"] = ann_constant_max

                    else:  # noqa: PLR5501
                        if window_data.isna().sum() == 0 and len(np.unique(window_data[~np.isnan(window_data)])) == 1:
                            fnl_df.loc[window_start:window_end, "ann_constant_long"] = ann_constant_max

        return fnl_df
