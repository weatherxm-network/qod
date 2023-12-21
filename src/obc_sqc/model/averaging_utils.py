from __future__ import annotations

import numpy as np
import pandas as pd


class AveragingUtils:
    """Functions for calculating the average of various meteorological variables."""

    @staticmethod
    def column_average_using_annotation(
        df: pd.DataFrame, column: str, availability_threshold: float, annotation_col: str
    ) -> float:
        """Calculates the average of column "column".

            Uses only non-faulty measurements according to the annotation in
            the "annotation_col" column.

        Args:
        ----
            df (DataFrame): the dataframe containing the data
            column (str): the name of the column which should be averaged
            availability_threshold (float): the percentage of available data (range 0 to 1), below which
                                            averaging or rewarding is not possible
            annotation_col (str): the name of the column which annotates with 0 all non-faulty data

        Returns:
        -------
            float: the corrected average of the given column
        """
        if len(df.index) == 0:
            return np.nan

        annotation_0_count: int = (df[annotation_col] == 0).sum()

        availability_percentage: float = annotation_0_count / len(df.index)

        if availability_percentage > availability_threshold:
            returned: float = df.loc[df[annotation_col] == 0, column].mean()
            return returned
        else:
            return np.nan

    @staticmethod
    def column_wind_speed_average_using_annotation(
        df: pd.DataFrame, availability_threshold: float, annotation_col: str
    ) -> float:
        """Calculates the wind speed vector average.

            Uses only non-faulty measurements according to the annotation in the
            "annotation_col" column. Columns "wind_u" and "wind_v" should be present
            in the provided dataframe.

        Args:
        ----
            df (DataFrame): the dataframe containing the data
            availability_threshold (float): the percentage of available data (range 0 to 1), below which
                                            averaging or rewarding is not possible
            annotation_col (str): the name of the column which annotates with 0 all non-faulty data

        Returns:
        -------
            float: the corrected wind speed vector average
        """
        if len(df.index) == 0:
            return np.nan

        annotation_0_count: int = (df[annotation_col] == 0).sum()

        availability_percentage: float = annotation_0_count / len(df.index)

        if availability_percentage > availability_threshold:
            u_avg: float = df.loc[df[annotation_col] == 0, "wind_u"].mean()
            v_avg: float = df.loc[df[annotation_col] == 0, "wind_v"].mean()

            wind_speed_avg: float = np.sqrt(u_avg**2 + v_avg**2)

            returned: float = round(wind_speed_avg, 2)

            return returned
        else:
            return np.nan

    @staticmethod
    def column_wind_direction_average_using_annotation(
        df: pd.DataFrame, availability_threshold: float, annotation_col: str
    ) -> float:
        """Calculates the wind direction vector average.

            Uses only non-faulty measurements according to the annotation in the
            "annotation_col" column. Columns "wind_u" and "wind_v" should be present
            in the provided dataframe.

        Args:
        ----
            df (DataFrame): the dataframe containing the data
            availability_threshold (float): the percentage of available data (range 0 to 1), below which
                                            averaging or rewarding is not possible
            annotation_col (str): the name of the column which annotates with 0 all non-faulty data

        Returns:
        -------
            float: the corrected wind direction vector average
        """
        if len(df.index) == 0:
            return np.nan

        annotation_0_count: int = (df[annotation_col] == 0).sum()

        availability_percentage: float = annotation_0_count / len(df.index)

        if availability_percentage > availability_threshold:
            u_avg: float = df.loc[df[annotation_col] == 0, "wind_u"].mean()
            v_avg: float = df.loc[df[annotation_col] == 0, "wind_v"].mean()

            wind_dir: float = np.degrees(np.arctan2(u_avg, v_avg))

            if wind_dir < 180:  # noqa: PLR2004
                wind_dir += 180
            elif wind_dir > 180:  # noqa: PLR2004
                wind_dir -= 180

            returned: float = round(wind_dir, 2)

            return returned
        else:
            return np.nan

    @staticmethod
    def row_wind_speed_calculation(row: pd.Series) -> float:
        """Calculates the wind speed from u and v components.

            Columns "u" and "v" should be present in the provided dataframe.

        Args:
        ----
            row (pd.Series): A pd.Series object containing columns "u" and "v"

        Returns:
        -------
            float: the calculated wind speed
        """
        wind_speed: float = np.sqrt(row["u"] ** 2 + row["v"] ** 2)

        return wind_speed

    @staticmethod
    def row_wind_direction_calculation(row: pd.Series) -> float:
        """Calculates the wind direction from u and v components.

            Columns "u" and "v" should be present in the provided dataframe.

        Args:
        ----
            row (pd.Series): A pd.Series object containing columns "u" and "v"

        Returns:
        -------
            float: the calculated wind direction
        """
        wind_dir: float = np.degrees(np.arctan2(row["u"], row["v"]))

        if pd.isna(wind_dir):
            return np.nan

        if wind_dir < 180:  # noqa: PLR2004
            wind_dir += 180
        elif wind_dir > 180:  # noqa: PLR2004
            wind_dir -= 180

        returned: float = round(wind_dir, 2)

        return returned
