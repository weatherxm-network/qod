from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    import pandas as pd


class FillingIgnoringPeriod:
    """Fill gaps in data, when the time gap is shorter than a specified period."""

    @staticmethod
    def filling_ignoring_period(
        fnl_df: pd.DataFrame, parameter: str, ignoring_period: int, data_timestep: int
    ) -> pd.DataFrame:
        """Fill gaps in data, when the time gap is shorter than a specified period.

        Creates an extra column with the values of the given parameter, but time gaps shorter than the
        ignoring period are filled with the last available observation. This allows us to ignore insignificant
        gap of data and forward the SQC process.

        Args:
        ----
            fnl_df (pd.DataFrame): the output of time_normalisation_dataframe(), which is a dataframe with a fixed
                        temporal resolution.
            parameter (str): the name of the parameter e.g., temperature, humidity, wind_speed etc.
            ignoring_period (int): the period within nans can be replaced with the latest valid value [in seconds]
            data_timestep (int): the desired timestep of the final df [in seconds]

        Returns:
        -------
            pd.DataFrame: a dataframe with all the parameters, but with one extra column for annotating out-of-bounds
                        values only for the selected parameter
        """
        rows_in_one_minute: int = int(
            round(ignoring_period / data_timestep)
        )  # how many rows are included in the ignoring_period
        mask: pd.Series = fnl_df[parameter].isna()  # masking for nan or no-nan values

        # The following process aims to fill with "fake" values all the single nan timeslots
        # and all the timeslots within the ignoring_period
        # So, at the end every ignoring_period or less will be filled with fake values (of the latest valid value).
        # Even if there is a blank period of 10 mins, the first "ignoring_period" seconds of this 10-min period
        # will be filled with a value.

        # Firstly, we need to fill with a value for single timeslots and set as nan consecutive blank timeslots.
        # Thus, [10,nan,10.2,10.3,10.4,10.5,nan,nan] equals to [10,10,10.2,10.3,10.4,10.5,nan,nan]
        # Fill single missing values with previous value e.g., [20,nan,20.1] = [20,20,20.1]
        fnl_df.loc[:, f"{parameter}_for_raw_check"] = fnl_df[parameter].fillna(method="ffill")

        # Check for consecutive missing values and fill with NaN
        consecutive_mask: pd.Series = mask & mask.shift(1)
        fnl_df.loc[consecutive_mask, f"{parameter}_for_raw_check"] = np.nan

        # Check for consecutive missing values within the ignoring_period and fill with previous valid number
        nans_in_last_minute: pd.Series = mask.rolling(rows_in_one_minute).sum()
        fnl_df[f"{parameter}_for_raw_check"] = fnl_df[f"{parameter}_for_raw_check"].fillna(
            method="ffill", limit=rows_in_one_minute
        )
        nans_in_last_minute_mask: pd.Series = (nans_in_last_minute > 0) & (nans_in_last_minute <= rows_in_one_minute)
        fnl_df.loc[nans_in_last_minute_mask, f"{parameter}_for_raw_check"] = fnl_df.loc[
            nans_in_last_minute_mask, f"{parameter}_for_raw_check"
        ].fillna(np.nan)

        # Check for consecutive missing values longer than 1 minute and fill with NaN
        nans_longer_than_one_minute_mask: pd.Series = nans_in_last_minute > rows_in_one_minute
        fnl_df.loc[nans_longer_than_one_minute_mask, f"{parameter}_for_raw_check"] = np.nan

        # Find the consecutive rows of filled values + an extra row, as we need to apply precipitation thresholds to the
        # next available value. This new column will be used by obc_precipitation
        mask = fnl_df[f"{parameter}"].isna() & fnl_df[f"{parameter}_for_raw_check"].notna()
        mask = mask | mask.shift(+1).fillna(False)

        # find the cumulative sum of False values across mask (every time we come across a False value
        # the cumulative sum increases, so we can distinct where a False value appears)
        # group the items based on the cumulative sum value (so each group contains max one False value)
        # find the cumulative sum of each group (the number of True values in the group)
        # Finally multiply by mask to zero-out the cumulative sum where the mask values are False
        fnl_df[f"{parameter}_consec_filling"] = mask.astype(int).groupby((~mask).cumsum()).cumsum() * mask

        return fnl_df
