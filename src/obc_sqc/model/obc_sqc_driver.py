from __future__ import annotations

import json

import pandas as pd

from obc_sqc.model.annotation_utils import AnnotationUtils
from obc_sqc.model.constant_data_check import ConstantDataCheck
from obc_sqc.model.filling_ignoring_period import FillingIgnoringPeriod
from obc_sqc.model.hour_averaging import HourAveraging
from obc_sqc.model.initial_params import InitialParams
from obc_sqc.model.minute_averaging import MinuteAveraging
from obc_sqc.model.raw_data_check import RawDataCheck
from obc_sqc.schema.schema import SchemaDefinitions


class ObcSqcCheck:
    """This class is the main class of the OBC/SQC algorithm."""

    @staticmethod
    def run(df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102, PLR0915, C901
        model: str = df["model"].iloc[0]

        (
            ann_unident_spk,
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
            raw_cntrl_thresholds,
            minute_cntrl_thresholds,
            minute_averaging_period,
            time_window_constant,
            obc_limits,
            start_timestamp,
            time_window_constant_max,
            ann_constant_max,
            pr_int,
            preprocess_time_window,
        ) = InitialParams.picking_initial_parameters(model)

        # Parameter to results
        results_mapping: dict[str, dict[str, pd.DataFrame]] = {}

        # loop through all parameters of a station
        for i, parameter in enumerate(parameters_for_testing):
            # Drop missing rows with missing weather data
            df.loc[
                df[SchemaDefinitions.weather_data_columns()].isna().sum(axis=1) > 0,
                SchemaDefinitions.weather_data_columns(),
            ] = pd.NA

            # Out of bounds check
            if parameter != "precipitation_accumulated":
                final_df: pd.DataFrame = ObcSqcCheck.obc(df, parameter, obc_limits[0][i], obc_limits[1][i])

            # Here we fill nans within the ignoring_period with previous available value, otherwise with nan
            final_df: pd.DataFrame = FillingIgnoringPeriod.filling_ignoring_period(
                final_df, parameter, ignoring_period, data_timestep
            )

            # Out of bounds check for precipitation must be applied after the filling_ignoring_period
            if parameter == "precipitation_accumulated":
                final_df: pd.DataFrame = ObcSqcCheck.obc_precipitation(final_df, obc_limits[0][i], obc_limits[1][i])

            if parameter != "precipitation_accumulated":
                # Shift all rows by 1 slot
                final_df["date"] = pd.to_datetime(final_df["utc_datetime"]) + pd.Timedelta(seconds=data_timestep)

                final_df_param = ConstantDataCheck.constant_data_check(
                    final_df,
                    parameter,
                    time_window_constant[i],
                    ann_constant,
                    ann_constant_frozen,
                    rh_threshold,
                    time_window_constant_max[i],
                    ann_constant_max,
                )
            else:
                final_df_param = final_df
                final_df_param["ann_constant"] = 0
                final_df_param["ann_constant_max"] = 0

            # Here, ONLY for WS2000 if wind speed is constantly at 0m/s for certain predefined period,
            # but wind direction varies, so that wind direction does not come with any of the constant annotations,
            # constant annotations are removed from wind speed too.
            if model == "WS2000":
                # Keep wind direction annotations to use them in the next iteration for wind speed
                if parameter == "wind_direction":
                    selected_columns = ["utc_datetime", "ann_constant", "ann_constant_long", "ann_constant_frozen"]
                    wdir_constant_df = final_df_param[selected_columns].copy()

                if parameter == "wind_speed":
                    merged_df = wdir_constant_df.merge(final_df_param, on="utc_datetime", suffixes=("_wdir", "_final"))
                    # Update 'ann_constant' in 'final_df_param' if wind direction is not constant
                    final_df_param.loc[merged_df["ann_constant_wdir"] == 0, "ann_constant"] = 0
                    final_df_param.loc[merged_df["ann_constant_long_wdir"] == 0, "ann_constant_long"] = 0
                    final_df_param.loc[merged_df["ann_constant_frozen_wdir"] == 0, "ann_constant_frozen"] = 0

            final_df_param = RawDataCheck.raw_data_suspicious_check(
                final_df_param,
                parameter,
                raw_cntrl_thresholds[i],
                data_timestep,
                time_window_median,
                availability_threshold_median[i],
                ann_unident_spk,
                ann_no_datum,
                ann_invalid_datum,
            )
            final_df_param = AnnotationUtils.text_annotation(final_df_param)

            # minute_averaging() can produce averages per minute (for WS1000) or per hour (for WS2000)
            final_df_param, minute_averaging = MinuteAveraging.minute_averaging(
                final_df_param,
                parameter,
                minute_averaging_period[i],
                availability_threshold_m[i],
                availability_threshold_median[i],
                time_window_median,
                minute_cntrl_thresholds[i],
                ann_invalid_datum,
                ann_unident_spk,
                pr_int,
                preprocess_time_window,
            )

            # keep only the useful columns in the raw result
            fnl_raw_process = final_df_param.loc[
                :,
                [
                    "utc_datetime",
                    parameter,
                    "rolling_median",
                    "consec_obs_diff_abs",
                    "median_diff_abs",
                    "ann_obc",
                    "ann_jump_couples",
                    "ann_invalid_datum",
                    "ann_unidentified_spike",
                    "ann_no_datum",
                    "ann_constant",
                    "ann_constant_long",
                    "ann_constant_frozen",
                    "total_raw_annotation",
                    "reward_annotation",
                    "annotation",
                ],
            ]

            results_mapping[parameter] = {
                "fnl_raw_process": fnl_raw_process,
                "hour_averaging": None,  # will be filled bellow
            }

            # Only stations with sampling rate <30sec can have both per minute and per hour checks
            if model == "WS1000":
                results_mapping[parameter]["hour_averaging"] = HourAveraging.hour_averaging(
                    minute_averaging,
                    fnl_timeslot,
                    availability_threshold_h[i],
                    parameter,
                )
            else:
                results_mapping[parameter]["hour_averaging"] = minute_averaging

            # calculate the hourly annotations (for both raw and minute-averaged data),
            # for the current parameter
            hourly_annotation: pd.Series = AnnotationUtils.error_codes_hourly(fnl_raw_process, minute_averaging)

            # assign the result to a new column, named "hourly_annotation", belonging to the "hour_averaging" key
            # of the results_mapping[parameter]
            df_to_modify: pd.DataFrame = results_mapping[parameter]["hour_averaging"]
            df_to_modify["hourly_annotation"] = hourly_annotation.apply(lambda x: x[0] + x[1])
            results_mapping[parameter]["hour_averaging"] = df_to_modify

        # Aggregate results
        param_items: list[pd.DataFrame] = []

        for p, v_dict in results_mapping.items():
            v = v_dict["hour_averaging"]

            def get_output(x: list[list]) -> str:
                """Extracts the annotations and percentages from a list.

                    The list should have the following
                    format: [['annotation1, percentage1'], ['annotation2, percentage2'], ...]

                Args:
                ----
                    x (list[list]): the list containing the ['annotation, percentage'] sublists

                Returns:
                -------
                    str: the json string containing the contents of the initial list
                """
                y: list = []
                if isinstance(x, list):
                    for sublist in x:
                        item: list = sublist[0].split(", ")
                        y.append([item[0], float(item[1])])
                json_str: str = json.dumps(y)
                return json_str

            # get the lists of hourly annotations from the first list of the v["hourly_annotation"] series,
            # which contains the annotations derived from raw data
            v["annotation"] = v["hourly_annotation"].apply(get_output)

            output_df: pd.DataFrame = v[["valid_percentage_rewards", "annotation"]].rename(
                columns={
                    "valid_percentage_rewards": f"{p}_score",
                    "annotation": f"{p}_annotation",
                }
            )
            param_items.append(output_df)

        result_df: pd.DataFrame = pd.concat(param_items, axis=1)

        # Aggregate results to a single DF
        flattened_results: dict[str, list[pd.DataFrame]] = {
            "fnl_raw_process": [],
            "hour_averaging": [],
        }

        total_rewards: float = ObcSqcCheck.calculate_daily_score(
            parameters_for_testing, results_mapping, flattened_results
        )
        qod_version: str = "1.0.6"

        result_df["qod_score"] = total_rewards
        result_df["hourly_score"] = result_df[[f"{x}_score" for x in results_mapping]].mean(axis=1)
        result_df["qod_version"] = qod_version
        final_df: pd.DataFrame = result_df.reset_index(names=["utc_datetime"])

        final_df["year"] = final_df["utc_datetime"].dt.year
        final_df["month"] = final_df["utc_datetime"].dt.month
        final_df["day"] = final_df["utc_datetime"].dt.day
        final_df["hour"] = final_df["utc_datetime"].dt.hour

        final_df.drop(columns=["utc_datetime"], inplace=True)  # noqa: PD002

        final_df_24h: pd.DataFrame = final_df.head(24)

        final_df_24h = ObcSqcCheck.daily_annotations(final_df_24h)
        return final_df_24h

    @staticmethod
    def daily_annotations(inp_df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        annotated_cols: list[str] = [
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "pressure",
            "illuminance",
            "precipitation_accumulated",
        ]
        df = inp_df.copy()

        # Daily annotations per weather variable
        daily_weather_ann: dict[str, [dict[str, float]]] = {}

        # All observed faults
        observed_faults: set[str] = set()

        # Produce daily annotations per weather variable
        for weather_col in annotated_cols:
            daily_annotation: dict[str, float] = {}
            for hourly_annotation in df[f"{weather_col}_annotation"]:
                for cur_fault, cur_perc in json.loads(hourly_annotation):
                    observed_faults.add(cur_fault)

                    if cur_fault not in daily_annotation:
                        daily_annotation[cur_fault] = 0

                    daily_annotation[cur_fault] += cur_perc / 24

            # Save for later
            daily_weather_ann[weather_col] = daily_annotation

            # Assign to all rows
            daily_anns: list[list[str | float]] = [[fault, score] for fault, score in daily_annotation.items()]
            df[f"daily_{weather_col}_annotation"] = json.dumps(daily_anns)

        # Produce a daily annotation for all weather vars by leveraging the per-weather variable annotations
        annotation: dict[str, list[list[str | float]]] = {fault: [] for fault in observed_faults}

        # Iterate over all faults and average over all weather columns
        for fault in observed_faults:
            fault_annotations: list[list[str | float]] = []

            # Average over all weather columns
            for weather_col in annotated_cols:
                daily_weather_col_ann: dict[str, float] = daily_weather_ann[weather_col]
                if fault in daily_weather_col_ann:
                    fault_payload: list[str | float] = [weather_col, daily_weather_col_ann[fault]]
                    fault_annotations.append(fault_payload)

            annotation[fault] = fault_annotations

        df["daily_annotation"] = json.dumps(annotation)
        return df

    @staticmethod
    def obc(fnl_df, parameter, bottom_lim, upper_lim):
        """This def annotates data as faulty when they exceed the manufacturer's limits

        fnl_df (df): the output of time_normalisation_dataframe def, which is a dataframe
            with a fixed temporal resolution
        parameter (str): the parameter that this def looks into, e.g., temperature, humidity, wind speed etc.
        bottom_lim (int): the bottom limit of the investigated parameter as defined by the manufacturer
        upper_lim (int): the upper limit of the investigated parameter as defined by the manufacturer

        result: a df with all the parameters, but with one extra column for annotating out-of-bounds values
                only for the selected parameter"""  # noqa: D202, D208, D209, D400, D415

        # Check if each element of a parameter is within the range defined by sensor's specs
        fnl_df["ann_obc"] = (
            ((fnl_df[parameter] < bottom_lim) | (fnl_df[parameter] > upper_lim)) & ~fnl_df[parameter].isna()
        ).astype(int)

        return fnl_df

    @staticmethod
    def obc_precipitation(fnl_df, bottom_lim, upper_lim):
        """This def annotates precipitation data as faulty when they exceed the manufacturer's limits.
        Precipitation comes as accumulation, so we de-accumulate it and then we apply OBC

        fnl_df (df): the output of time_normalisation_dataframe def, which is a dataframe with a fixed
            temporal resolution
        bottom_lim (int): the bottom limit of the investigated parameter as defined by the manufacturer
        upper_lim (int): the upper limit of the investigated parameter as defined by the manufacturer

        result: a df with all the parameters, but with one extra column for annotating out-of-bounds values
            only for the selected parameter
        """  # noqa: D202, D205, D400, D415

        # Calculate the difference between current and previous elements
        fnl_df["precipitation_diff"] = fnl_df["precipitation_accumulated_for_raw_check"].diff()

        # Set values of 'ann_obc' based on the condition
        fnl_df["ann_obc"] = 0  # Initialize 'ann_obc' column with zeros

        fnl_df.loc[
            (fnl_df["precipitation_diff"] > (upper_lim * (fnl_df["precipitation_accumulated_consec_filling"] + 1)))
            | (fnl_df["precipitation_diff"] < bottom_lim),
            "ann_obc",
        ] = 1

        return fnl_df

    @staticmethod
    def calculate_daily_score(parameters_for_testing, results_mapping, flattened_results) -> float:  # noqa: D102
        for param in parameters_for_testing:
            for key, value in results_mapping[param].items():
                value["parameter"] = param
                flattened_results[key].append(value)

        rewards_per_parameters: dict[str, pd.Series] = {
            parameters_for_testing[i]: value["valid_percentage_rewards"].to_list()
            for i, value in enumerate(flattened_results["hour_averaging"])
        }

        adjusted_hourly_rewards: dict[str, list[float]] = {p: [] for p in parameters_for_testing}
        for p, param_vals in rewards_per_parameters.items():
            for hourly_reward_percentage in param_vals:
                reward = hourly_reward_percentage / 100
                adjusted_hourly_rewards[p].append(reward)

        adjusted_daily_rewards: dict[any, float] = {}
        for param_name, param_vals in adjusted_hourly_rewards.items():
            potential_daily_rewards: float = sum(param_vals) / len(param_vals)
            adjusted_daily_rewards[param_name] = potential_daily_rewards

        # Final single score for rewards
        total_rewards: float = sum(adjusted_daily_rewards.values()) / len(adjusted_daily_rewards)
        return total_rewards
