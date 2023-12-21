from __future__ import annotations

from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec
from mlflow.types import DataType
from mlflow.types import Schema


class SchemaDefinitions:  # noqa: D101
    @staticmethod
    def mlflow_obc_sqc_schema():  # noqa: D102
        return {
            "temperature_score": "Float64",
            "humidity_score": "Float64",
            "wind_speed_score": "Float64",
            "wind_direction_score": "Float64",
            "pressure_score": "Float64",
            "illuminance_score": "Float64",
            "precipitation_accumulated_score": "Float64",
            "temperature_annotation": str,
            "humidity_annotation": str,
            "wind_speed_annotation": str,
            "wind_direction_annotation": str,
            "pressure_annotation": str,
            "illuminance_annotation": str,
            "precipitation_accumulated_annotation": str,
            "daily_temperature_annotation": str,
            "daily_humidity_annotation": str,
            "daily_wind_speed_annotation": str,
            "daily_wind_direction_annotation": str,
            "daily_pressure_annotation": str,
            "daily_illuminance_annotation": str,
            "daily_precipitation_accumulated_annotation": str,
            "daily_annotation": str,
            "model": str,
            "qod_score": "Float64",
            "hourly_score": "Float64",
            "qod_version": str,
            "year": "Int64",
            "month": "Int64",
            "day": "Int64",
            "hour": "Int64",
        }

    @staticmethod
    def mlflow_signature():  # noqa: D102
        # Log model for future use
        input_schema = Schema(
            [
                ColSpec(DataType.string, "utc_datetime"),
                ColSpec(DataType.double, "temperature"),
                ColSpec(DataType.double, "humidity"),
                ColSpec(DataType.double, "wind_speed"),
                ColSpec(DataType.double, "wind_direction"),
                ColSpec(DataType.double, "pressure"),
                ColSpec(DataType.double, "illuminance"),
                ColSpec(DataType.double, "precipitation_accumulated"),
                ColSpec(DataType.string, "model"),
            ]
        )

        output_schema = Schema(
            [
                ColSpec(DataType.double, "temperature_score"),
                ColSpec(DataType.double, "humidity_score"),
                ColSpec(DataType.double, "wind_speed_score"),
                ColSpec(DataType.double, "wind_direction_score"),
                ColSpec(DataType.double, "pressure_score"),
                ColSpec(DataType.double, "illuminance_score"),
                ColSpec(DataType.double, "precipitation_accumulated_score"),
                ColSpec(DataType.string, "temperature_annotation"),
                ColSpec(DataType.string, "humidity_annotation"),
                ColSpec(DataType.string, "wind_speed_annotation"),
                ColSpec(DataType.string, "wind_direction_annotation"),
                ColSpec(DataType.string, "pressure_annotation"),
                ColSpec(DataType.string, "illuminance_annotation"),
                ColSpec(DataType.string, "precipitation_accumulated_annotation"),
                ColSpec(DataType.string, "daily_temperature_annotation"),
                ColSpec(DataType.string, "daily_humidity_annotation"),
                ColSpec(DataType.string, "daily_wind_speed_annotation"),
                ColSpec(DataType.string, "daily_wind_direction_annotation"),
                ColSpec(DataType.string, "daily_pressure_annotation"),
                ColSpec(DataType.string, "daily_illuminance_annotation"),
                ColSpec(DataType.string, "daily_precipitation_accumulated_annotation"),
                ColSpec(DataType.string, "daily_annotation"),
                ColSpec(DataType.string, "model"),
                ColSpec(DataType.double, "qod_score"),
                ColSpec(DataType.double, "hourly_score"),
                ColSpec(DataType.string, "qod_version"),
                ColSpec(DataType.integer, "year"),
                ColSpec(DataType.integer, "month"),
                ColSpec(DataType.integer, "day"),
                ColSpec(DataType.integer, "hour"),
            ]
        )

        params = ParamSchema(
            [
                ParamSpec("log_to_opensearch", "boolean", False),
                ParamSpec("device_id", "string", "UNKNOWN"),
            ]
        )

        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params)
        return signature

    @staticmethod
    def qod_input_schema():  # noqa: D102
        return {
            "temperature": "Float64",
            "humidity": "Float64",
            "wind_speed": "Float64",
            "wind_direction": "Float64",
            "pressure": "Float64",
            "illuminance": "Float64",
            "precipitation_accumulated": "Float64",
            "model": str,
            "utc_datetime": str,
        }

    @staticmethod
    def weather_data_columns():  # noqa: D102
        return [
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "pressure",
            "illuminance",
            "precipitation_accumulated",
        ]
