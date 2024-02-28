from __future__ import annotations

import numpy as np
import pandas as pd


class RawDataChecks:  # noqa: D101
    @staticmethod
    def raw_data_suspicious_check(  # noqa: PLR0915
        fnl_df,
        parameter,
        control_threshold,
        minimum_timestep,
        time_window,
        availability_threshold_median,
        ann_unident_spk,
        ann_no_datum,
        ann_invalid_datum,
    ):
        """Detects faulty observations based on WMO criteria.

        Furthermore, it annotates as faulty the observation which is unavailable or has the greatest difference
        from the 10-minute median in case of a significant jump within a couple of raw data.
        As an example, if we have a dataset of temperatures [10,10,10.1,50,10.1,50,50,50,10.2, 10.2,nan,10.2],
        then e.g., the difference between the 3rd and 4th elements is >2, but the difference of the
        4th element from the 10-min median is larger, and thus the 4th element is faulty. Also 6th, 7th and 8th
        elements will be considered as faulty because  they are equal to the latest faulty measurement. In addition,
        the nan value will be annotated as faulty/invalid. It should be noted that if median cannot be calculated due to
        significant amount of unavailable data, then all the suspicious for spike/dip values will be annotated
            as faulty.
        fnl_df (df): the output of time_normalisation_dataframe def, which is a dataframe with a fixed temporal
        resolution parameter (str): the parameter that this def looks into, e.g., temperature, humidity, wind speed etc.
        control_threshold (int): the threshold to check for jumps in a parameter
        minimum_timestep (int): the timestep of the reframed raw data (e.g., 16sec) [in seconds]
        time_window (int): the time window for rolling median [in minutes]
        availability_threshold_median (float): the availability threshold, e.g., we are able to calculate median
            only if <67%/75% of timeslots within a certain period are available
        ann_unident_spk (int): annotation where no median has been calculated
        ann_no_datum (int): annotation where no datum is available
        ann_invalid_datum (int): annotation where the datum exists, but it's not valid

        result (df): a df with the
            a. parameter under investigation,
            b. rolling median,
            c. abs difference between consecutive observations,
            d. abs difference between measurement and last 10-min median,
            e. annotation for couples of invalid jump, invalid datum (the one obs from a couple),
                not available median, not available datum, constant data, unidentified spike
                (for both all- and -reward faulty data)
            f. text annotations for all reasons that a datum is faulty
        """  # noqa: D202

        # Sort the DataFrame by date
        fnl_df = fnl_df.sort_values("utc_datetime")

        # Set the date column as the index
        if fnl_df.index.name is None:
            # Ensure that UTC datetime is of type 'datetime64[ns]', otherwise the 'rolling' fails downstream
            fnl_df["utc_datetime"] = pd.to_datetime(fnl_df["utc_datetime"])
            fnl_df.set_index("utc_datetime", inplace=True)  # noqa: PD002

        # We exclude the parameter of wind direction. No hump check can be applied on wind direction.
        if parameter != "wind_direction" and parameter != "precipitation_accumulated":  # noqa: PLR1714
            # Calculating the 10-min rolling median
            rolling_median = fnl_df[f"{parameter}_for_raw_check"].rolling(f"{time_window}min").median()

            # Getting the number of available observations within the time_window
            available_observations = fnl_df[f"{parameter}_for_raw_check"].rolling(f"{time_window}min").count()

            # Calculating the total number of possible observations within the time_
            # window (e.g., 10 minutes / 16 seconds)
            possible_observations = time_window * 60 / minimum_timestep

            # Finding the percentage of the available observations
            percentage_available = available_observations / possible_observations

            # Assign np.nan to rolling median where percentage of available observations is
            # less than the availability_threshold_median (e.g., 67%)
            rolling_median[percentage_available < availability_threshold_median] = np.nan

            # Create a new column in the df with the rolling median values
            fnl_df["rolling_median"] = rolling_median

            # Get the absolute difference between consecutive observations
            fnl_df["consec_obs_diff_abs"] = fnl_df[f"{parameter}_for_raw_check"].diff().abs()

            # Finding the absolute difference between an observation and the median of the last x minutes
            fnl_df["median_diff_abs"] = (fnl_df[f"{parameter}_for_raw_check"] - fnl_df["rolling_median"]).abs()

            # Annotate both the values of a couple of observations with 1 and finally
            # annotate which values has a larger difference from median
            # Create 4 columns to use them for annotation
            fnl_df["ann_jump_couples"] = 0  # for annotating both observations in case of an invalid jump

            # for annotating only one of the observations of a couple with an invalid jump
            fnl_df["ann_invalid_datum"] = 0

            # if median cannot be calculated and there are suspicious for spike/dip values
            fnl_df["ann_unidentified_spike"] = 0

            # if there is no datum in a certain timeslot
            fnl_df["ann_no_datum"] = 0

            temp = fnl_df[f"{parameter}_for_raw_check"]
            fnl_df.reset_index(inplace=True)  # noqa: PD002

            for i in range(1, len(temp)):
                prev_val = fnl_df.loc[i - 1, "median_diff_abs"] if i > 0 else 0
                curr_val = fnl_df.loc[i, "median_diff_abs"]
                prev_or_cur_val_is_missing: bool = prev_val is pd.NA or curr_val is pd.NA

                # check if difference is larger than the defined threshold
                diff = abs(temp[i] - temp[i - 1]) if temp[i] is not pd.NA and temp[i - 1] is not pd.NA else pd.NA
                if not prev_or_cur_val_is_missing:
                    if diff > control_threshold:
                        fnl_df.loc[i, "ann_jump_couples"] = 1
                        fnl_df.loc[i - 1, "ann_jump_couples"] = 1

                        # In case the difference is large, check which abs(value-median)
                        # of the couple of observations is larger and annotate
                        if prev_val > curr_val:
                            fnl_df.loc[i - 1, "ann_invalid_datum"] = ann_invalid_datum
                        elif curr_val > prev_val:
                            fnl_df.loc[i, "ann_invalid_datum"] = ann_invalid_datum

                else:
                    fnl_df.loc[i, "ann_jump_couples"] = 0
                    fnl_df.loc[i, "ann_invalid_datum"] = 0

                # Annotating as invalid, observations that are equal to a previous invalid value
                if not prev_or_cur_val_is_missing:
                    if temp[i] == temp[i - 1] and fnl_df["ann_invalid_datum"][i - 1] == ann_invalid_datum:
                        fnl_df.loc[i, "ann_invalid_datum"] = ann_invalid_datum
                else:
                    fnl_df.loc[i, "ann_invalid_datum"] = 0

            no_median_mask = fnl_df["rolling_median"].isnull()  # noqa: PD003
            ann_jump_couples_mask = fnl_df["ann_jump_couples"] == 1
            combined_mask = no_median_mask & ann_jump_couples_mask

            # Annotate for 'ann_unidentified_spike' where 'median_diff_abs' is null, but there are unexpected spikes
            fnl_df.loc[combined_mask, "ann_unidentified_spike"] = ann_unident_spk

        # As this series of checks is not for wind direction, all the
        # following columns are created just for facilitating the algorithm
        else:
            fnl_df.reset_index(inplace=True)  # noqa: PD002
            fnl_df["rolling_median"] = np.nan
            fnl_df["consec_obs_diff_abs"] = np.nan
            fnl_df["median_diff_abs"] = np.nan
            fnl_df["ann_jump_couples"] = np.nan
            fnl_df["ann_invalid_datum"] = np.nan
            fnl_df["ann_unidentified_spike"] = np.nan
            fnl_df["ann_no_datum"] = 0

        # Create a boolean mask where missing values are marked as True. So, unavailable data are annotated.
        no_observation_mask = fnl_df[parameter].isnull()  # noqa: PD003
        fnl_df.loc[no_observation_mask, "ann_no_datum"] = ann_no_datum

        # Make a new column where all faulty elements (for any reason)
        # detected in previous processes are annotated with 1.
        # Thus, we merge all faults in a new column.
        fnl_df["total_raw_annotation"] = np.where(
            (fnl_df["ann_obc"] > 0)
            | (fnl_df["ann_invalid_datum"] > 0)
            | (fnl_df["ann_unidentified_spike"] > 0)
            | (fnl_df["ann_no_datum"] > 0)
            | (fnl_df["ann_constant_frozen"] > 0)
            | (fnl_df["ann_constant"] > 0)
            | (fnl_df["ann_constant_long"] > 0),
            1,
            0,
        )

        # Make a new column where only reward-faulty elements are annotated with 1.
        fnl_df["reward_annotation"] = np.where(
            (fnl_df["ann_obc"] > 0)
            | (fnl_df["ann_invalid_datum"] > 0)
            | (fnl_df["ann_unidentified_spike"] > 0)
            | (fnl_df["ann_no_datum"] > 0)
            | (fnl_df["ann_constant_frozen"] > 0)
            | (fnl_df["ann_constant"] > 0)
            | (fnl_df["ann_constant_long"] > 0),
            1,
            0,
        )

        fnl_df.set_index("utc_datetime", inplace=True)  # noqa: PD002

        
        return fnl_df
