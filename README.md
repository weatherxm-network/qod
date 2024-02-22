# WeatherXM QoD

## Intro

WeatherXM's QoD (Quality-of-Data) is the algorithm that evaluates the quality of weather data provided by the network's weather stations. The calculated QoD score indicates the confidence level in the quality of the weather data received from a station. We need meaningful and usable data and the QoD score is an attempt to quantify this metric. The end goal is to encourage weather station owners to do their best to comply with the Network's guidelines in order to consistently achieve the best QoD score possible.

The QoD score of each station is an important parameter that controls its daily rewards, along with a few other criteria. Read more about WeatherXM's reward scheme [here](https://docs.weatherxm.com/reward-mechanism).

## How it works

QoD involves a series of techniques and processes designed to help us distinguish between expected and unexpected data behaviors. This mechanism includes the following 3 control points and their respective checks:

1. Self Check
   - Out-of-Bounds Check (OBC) (since QoD v1.0)
   - Self Quality Check (SQC) (since QoD v1.0)

2. Reference Check
   - Comparative Quality Control (CQC) (coming in QoD v1.3)

3. Deployment status
   - Indoor Station Detector (ISD) (coming in QoD v1.1)
   - Solar Obstacle Detector (SOD) (coming in QoD v1.2)
   - Wind Obstacle Detector (WOD) (TBD)
  
Read more about QoD and our rewards program on [docs.weatherxm.com](https://docs.weatherxm.com/project).

A more detailed description of the QoD algorithm is available in section [Quality of Data (QoD) Mechanism v1 - Full Description](#quality-of-data-qod-mechanism-v1---full-description)

## Prerequisites

To run the QoD algorithm for a specific date you you will need to provide the data for this and the previous day. Datasets have been upload to Filecoin via [Basin](https://github.com/tablelandnetwork/basin-cli) under the publication `w_tst1.data` (the publication will change for production) by running `basin publication deals --publication w_tst1.data`. You can find all available data using Basin's CLI and running `basin publication deals --publication w_tst1.data`.

## Running on Docker (locally)

```bash
docker build -t wxm-qod:local .

docker run \
	-v /datasets:/datasets \
	wxm-qod:local obc_sqc.iface.file_model_inference\
	--device_id <device_id> \
	--date <date> \
	--day1 <yesterday> \
	--day2 <today> \
	--output_file_path <path>
```

### Parameters
- `--device_id`: The device ID for which QoD will be calculated
- `--date`: The to calculate QoD for
- `--day1`: Path pointing to the data for the day before the one QoD will be calculated for
- `--day2`: Path pointing to the data for the day for which QoD will be calculated
- `--output_file_path`: Path pointing to the file where the results will be written at

### Example

To calculate the QoD for `2023-12-14` for the device `8bf5b8f0-50db-11ed-960b-b351f0b0cc44` we need to download the datasets for `2023_12_13` and `2023_12_14` and save them under the same folder (the example below assumes you will save them under `/datasets` and name them `2023_12_13.parquet` and `2023_12_14.parquet` respectively)

```bash
docker run
	-v /datasets:/datasets
	--device_id 8bf5b8f0-50db-11ed-960b-b351f0b0cc44 \
	--date "2023-12-14" \
	--day1 /datasets/2023_12_13.parquet \
	--day2 /datasets/2023_12_14.parquet \
	--output_file_path /outputs/result
	wxm-qod:local
```

## Running on Bacalhau

### Requirements
1. Install Bacalhau based on the [instructions](https://docs.bacalhau.org/getting-started/installation)
2. Find the CIDs of the datasets for the days you want to run the QoD on https://index.weatherxm.network/

### Run the Bacalhau job

```bash
bacalhau docker run \
  --memory 32GB \
  --cpu 12 \
  -i src=ipfs://bafybeiagzaeyk2cy4bjp5yiejthuvfiulbo34urbqvewvhqevxusg2cfce,dst=/input_2024_02_14.parquet \
  -i src=ipfs://bafybeia3hbh5eek4vuq7wqe3xwlmtetjmsfyml37q5fwf2pk73uu63es3a,dst=/input_2024_02_15.parquet \
  ghcr.io/weatherxm-network/qod:0.0.1 -- \
  obc_sqc.iface.file_model_inference \
  --device_id 8bf5b8f0-50db-11ed-960b-b351f0b0cc44 \
  --date "2024-02-15" \
  --day1 /input_2024_02_14.parquet \
  --day2 /input_2024_02_15.parquet \
  --output_file_path /outputs/result
```

You can see an example out put by fetching the results:
```bash
bacalhau get bf059011-e744-40a0-9145-137d6e0803e4
```

## Quality of Data (QoD) Mechanism v1 - Full Description

QoD serves as the mechanism used to differentiate between accurate and erroneous data recorded by a weather station. To achieve this, we employ a series of techniques and processes that scrutinise different aspects related to data quality.

The current iteration of QoD incorporates control checks to pinpoint instances where a sensor records values beyond the limits specified by the manufacturer, identifies unusual jumps, recognises constant values, or determines if there is insufficient data to compute an average.

The processes included in the QoD mechanism v1 are divided into two categories:
- Out-of-Bounds Check (OBC)
- Self Quality Check (SQC)

QoD v1 is able to process:
- Time series with an interval <30 seconds (High Temporal Resolution Time series - **HTRT**)
- Time series with a time interval ≥1 minute (Low Temporal Resolution Time series - **LTRT**). We consider that the value provided on this time interval (e.g., 3 minutes) time interval is the outcome derived from the averaging of a sequence of internally recorded measurements.

The **final output of the QoD mechanism** is an hourly **percentage** of valid/available data and the corresponding **text annotations per meteorological variable** (temperature, relative humidity, wind speed, wind direction, atmospheric pressure, illuminance and precipitation). However, there is a post QoD score process that aggregates all the annotations and scores into a daily output per station.

Before moving forward to the description of the QoD mechanism, we should note that the input data has been standardised over time (resampling). Weather stations may transmit data at varying non-fixed time intervals. This implies that even if a station's specification indicates a recording frequency of 16 seconds, the actual interval can fluctuate, ranging from a few seconds to several minutes. To ensure that the quality control operates on a time-normalised dataset, a new timeframe is established with a fixed interval (determined by the theoretical recording frequency (f<sub>r</sub>) of the station model, such as 16" for the WG1000 - M5 model).

### 1. Filling Ignoring period - Filling No-Data Slots

To ensure the proper functioning of certain SQC processes, an ignoring time period is established [Table 1](#table-1---ignoring-period-for-the-available-weather-station-models). In practice, all time slots within the ignoring period that lack data are filled with the most recent valid value. Subsequently, this newly filled time series is exclusively employed for scrutinising constant data and anomalous jumps at the raw scale level (see [section 3](#3-self-quality-check-sqc)).

##### Table 1 - Ignoring period for the available weather station models.

| Weather Station Model | Ignoring Period [seconds] |
|------------------------|---------------------------|
| WG1000 (M5)            | 60                        |
| WS2000 (Helium)        | 180                       |

### 2. Out-of-Bounds Check (OBC)

OBC is a simple process aimed at identifying values for each parameter that exceed the limits set by the manufacturer. The OBC proves highly valuable in identifying sensors that might be experiencing technical malfunctions. OBC is a simple check for temperature, relative humidity, wind speed/direction, atmospheric pressure, illuminance and precipitation that stands alone.

Particularly for the parameter of precipitation, OBC applies to the variable "accumulated precipitation". As OBC for precipitation looks at the accumulated precipitation difference between two consecutive timesteps, it should be applied over the filled time series produced in [section 1](#1-filling-ignoring-period---filling-no-data-slots). This allows for a discrimination of faulty precipitation rates, even under the lack of data for a predefined short period of time. The upper cut-off limit used for this process is time dependent, as its default unit is in mm/sec. Thus, this limit is adjusted depending on the recording frequency of the station model (e.g., from 0.254mm/sec to 4.064mm/16sec) and, eventually, the OBC is applied on the difference of the "accumulated precipitation". In case of a no-data slot that has been filled (according to the process in [section 1](#1-filling-ignoring-period---filling-no-data-slots)), the precipitation cut-off threshold is reduced to the corresponding time (e.g., for a data temporal resolution of 16", if time slots at 16", 32" and 48" have no data and have been filled with the value of the latest valid measurement at 00", and the next available measurement is at 64", the cut-off limit will gradually increase for each filled timestep till the next valid datum, such as 4.064, 8.128, 12.192, 16.256mm).

Currently, there are two sets of outdoor sensors (WS1000 and WS2000), each accompanied by its own respective specifications outlined in [Table 2](#table-2---manufacturers-limits-for-ws1000-m5-sensors) and [Table 3](#table-3---manufacturers-limits-for-ws2000-helium-sensors).

##### Table 2 - Manufacturer's limits for WS1000 (M5) sensors.

| Parameter                  | Lower Limit | Upper Limit |
|----------------------------|-------------|-------------|
| Temperature [°C]           | -40         | 60          |
| Relative Humidity [%]      | 10          | 99          |
| Wind Direction [°]         | 0           | 359         |
| Wind Speed [m/s]           | 0           | 50          |
| Atmospheric Pressure [mb]  | 300         | 1100        |
| Illuminance [Lux]          | 0           | 400000      |
| Precipitation [mm]         | 0           | 0.254/sec   |

<br>

##### Table 3 - Manufacturer's limits for WS2000 (Helium) sensors.

| Parameter                  | Lower Limit | Upper Limit |
|----------------------------|-------------|-------------|
| Temperature [°C]           | -40         | 80          |
| Relative Humidity [%]      | 1           | 99          |
| Wind Direction [°]         | 0           | 359         |
| Wind Speed [m/s]           | 0           | 50          |
| Atmospheric Pressure [mb]  | 540         | 1100        |
| Illuminance [Lux]          | 0           | 200000      |
| Precipitation [mm]         | 0           | 0.254/sec   |

### 3. Self Quality Check (SQC)

SQC is a process that aims to detect anomalous behaviour of weather observations due to faulty sensors or not proper deployments using only the data of a single weather station itself. The thresholds, but also the assumptions that lead to the final result, are based on the [WMO](https://library.wmo.int/doc_num.php?explnum_id=4236) and [European Commission](https://publications.jrc.ec.europa.eu/repository/handle/JRC128152) general guidelines. However, our further data analysis based on data from the WeatherXM network results in corrected and less strict thresholds at some SQC checks.

As there are weather stations that transmit data at various time intervals, ranging from a few seconds to several minutes, SQC should be capable of annotating data regardless of the temporal resolution of a data time series. However, it is important to highlight that temporal resolutions exceeding 1' may not be suitable for achieving rigorous control checks.

SQC consists of a series of controls per parameter (temperature, relative humidity, wind speed, wind direction, atmospheric pressure and illuminance) at different time scales, which differ for different temporal resolutions (HTRT, LTRT). Note that precipitation is excluded from SQC, as it requires a careful and more complex quality control.

A HTRT passes through ([Table 4](#table-4---all-the-control-processes-that-extend-across-the-three-time-scale-levels-for-the-case-of-a-weather-station-that-records-data-every-30-seconds-or-less-eg-wg1000-m5)):
<ol type="a">
  <li>raw scale level control (every single of the raw data is checked for suspicious jumps, unavailable and constant data),</li>
  <li>minute scale level control (1-minute averaged data are checked for suspicious jumps and unavailable data)</li>
  <li>hourly scale level control (final counting of invalid/unavailable hourly time slots)</li>
</ol>

For LTRT, practically only the raw (which is now called inter-minute scale level) and the hourly scale levels are applied, as the minute averaging is not applicable ([Table 5](#table-5---all-the-control-processes-that-extend-across-the-two-time-scale-levels-for-the-case-of-a-weather-station-that-records-data-every-1-minute-or-more-eg-ws2000-helium)).

##### Table 4 - All the control processes that extend across the three time scale levels for the case of a weather station that records data every 30 seconds or less (e.g., WG1000 (M5)).

<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    text-align: center;
    border: 1px solid black;
    padding: 8px;
  }
</style>

<table>
  <tr>
    <th colspan="7">Control Process of HTRTs (&lt;30sec)</th>
  </tr>
  <tr>
    <th>Time Scale</th>
    <td colspan="3">Raw Scale</td>
    <td colspan="2">Minute Scale</td>
    <td>Hourly Scale</td>
  </tr>
  <tr>
    <th>Control Check Type</th>
    <td>Constant data detection</td>
    <td>Unavailable data detection</td>
    <td>Suspicious Jump Detection</td>
    <td>Unavailable data detection</td>
    <td>Suspicious Jump Detection</td>
    <td>Counting of invalid hourly time slots</td>
  </tr>
</table>

<br>

##### Table 5 - All the control processes that extend across the two time scale levels for the case of a weather station that records data every 1 minute or more (e.g., WS2000 (Helium)).

<table>
  <tr>
    <th colspan="5">Control Process of LTRTs (&gt;30sec)</th>
  </tr>
  <tr>
    <th>Time Scale</th>
    <td colspan="3">Inter-Minute Scale</td>
    <td>Hourly Scale</td>
  </tr>
  <tr>
    <th>Control Check Type</th>
    <td>Constant data detection</td>
    <td>Unavailable data detection</td>
    <td>Suspicious Jump Detection</td>
    <td>Counting of invalid hourly time slots</td>
  </tr>
</table>

There are three basic processes to check **Constancy**, **Availability** and **Fluctuation** of the data, these are **a. Constant data detection**, **b. Unavailable data detection** and **c. Suspicious jump detection**.

#### a. Constancy Checks

The constant data detection process aims to identify whether a sensor is recording unchanging values due to technical disruptions or improper deployment. This process is applied to raw data to ensure that even insignificant alterations over a span of a few seconds are taken into account. According to WMO recommendations, the parameters of temperature, relative humidity, wind speed, wind direction and atmospheric pressure should not remain constant for more than 1 hour. However, our current approach employs a more flexible strategy, employing different thresholds according to our data analysis.

While constant data typically raises suspicion, there are certain cases that warrant exclusion. For instance, during foggy or fairly wet conditions, there is often a prolonged period of constant relative humidity/temperature/wind spanning hours, and thus, such cases are excluded from this process ([Table 6](#table-6---duration-thresholds-for-the-constant-data-detection-process) - Constancy Duration Threshold). Regarding wind speed, constant, but non-zero wind speed under freezing temperatures is considered as faulty (due to faulty sensor). Note that in the particular case of WS2000, constant annotations on wind speed measurements are removed if wind direction comes with no constant annotations (this applies to all types of constant annotations). Regarding illuminance, we establish a maximum threshold for constant illuminance when it is not equal to 0 lux. This is because it may remain at ≠0Lux for extended hours during the night, but it should not remain constant at a specific value for an extended period during the day.

Finally, a second set of constancy checks is established to mainly identify faulty sensors ([Table 6](#table-6---duration-thresholds-for-the-constant-data-detection-process) - Constancy Duration Max Threshold). Consequently, any time series under investigation is marked as erroneous when the data remain constant for 1440 minutes without any additional specific conditions. It is worth noting that relative humidity has been excluded from this criterion due to its potential to remain constant for longer durations. In addition, illuminance and pressure are also excluded since the initial set of criteria (with a 120-minute threshold) adequately addresses constancy checks for this parameter.

##### Table 6 - Duration thresholds for the constant data detection process.

| Variable             | Constancy Duration Threshold (minutes) | Constancy Duration Max Threshold (minutes) | Control Checks (annotated as faulty) | Exclusions |
|----------------------|----------------------------------------|--------------------------------------------|----------------------------------------|------------|
| Temperature          | 240                                    | 1440                                       | Constant values when RH<sub>median</sub><95%                                                                                                                                                                                              | Constant values when RH<sub>median</sub>≥95% |
| Relative Humidity    | 360                                    | -                                          | Constant values when RH<sub>median</sub><95%                                                                                                                                                                                              | Constant values when RH<sub>median</sub>≥95% |
| Wind Speed           | 360                                    | 1440                                       | - Constant wind speed to 0m/s when T<sub>median</sub> is >0°C <br> - Constant values when Tmedian is <0°C (annotated as frozen) <br> - Constant wind speed to 0m/s when RH<sub>median</sub><85% <br> - Constant wind speed to value ≠0m/s | Constant values when RH<sub>median</sub>≥85% |
| Wind Direction       | 360                                    | 1440                                       | - Constant wind direction when T<sub>median</sub> is >0°C <br> - Constant wind direction when RH<sub>median</sub><85%                                                                                                                     | - Constant values when T<sub>median</sub> is ≤0°C <br> - Constant values when RH<sub>median</sub>≥85% |
| Atmospheric Pressure | 120                                    | -                                          | Constant pressure regardless the pressure value                                                                                                                                                                                           |
| Illuminance          | 120                                    | -                                          | Constant illuminance only when it is ≠0Lux | Constant illuminance only when it is 0Lux

#### b. Unavailable Data Detection

Having already resampled the timeframe, we then check for no-data timeslots. We count the no-data time slots at both raw and minute scale levels. Note that in minute scale level, a no-data timeslot may arise due to lack of raw data within a certain minute or invalidity of data (e.g., see [section 3c](#c-suspicious-jump-detection)).

In order to be able to produce 1-10min averages (as [WMO recommends](https://library.wmo.int/viewer/41650/download?file=8_III_2021_en.pdf&type=pdf&navigator=1)), one valid (which has passed all the previous control checks) data package per minute and one per 3 minutes is required for HTRTs and LTRTs respectively. Note that in both cases a data package is the result of averaging measurements within that period of time (which coincides with the uplink rate of each weather station e.g., 16" and 3' for WS1000 and WS2000). However, in cases of weather stations at remote places with difficulties in connection and power, even a single data package per hour is still valuable (e.g., comparing with numerical model outputs) and should be scored by the QoD mechanism. Finally, no-data slots are counted in the Minute/Inter-Minute level (HTRTs/LTRTs) and the hourly result eventually contributes to the final QoD score.

To generate 1-10 minute averages, as recommended by [the WMO](https://library.wmo.int/viewer/41650/download?file=8_III_2021_en.pdf&type=pdf&navigator=1), a valid data package is required for each minute (for HTRTs) and one every three minutes (for LTRTs). A data package, resulting from averaged measurements within the specified time period, aligns with the uplink rate of each weather station, such as 16" and 3' for WS1000 and WS2000, respectively. It's important to note that, in instances where weather stations face challenges in connectivity and power (e.g., stations on mountain tops), even a single data package per hour holds value (e.g., for comparison with numerical model outputs) and should be assessed by the QoD mechanism. Additionally, no-data slots are considered at the Minute/Inter-Minute level (HTRTs/LTRTs), and the hourly outcome ultimately contributes to the final QoD score.

#### c. Suspicious Jump Detection

Suspicious jump detection is employed to detect abrupt and unusual changes in a parameter occurring within a short period of time. Such changes are often attributed to technical disruptions, although in fewer instances, they may result from improper deployment. This process is applied along both raw and minute scale levels for HTRTs ([Table 7](#table-7---jump-thresholds-for-the-raw-and-minute-scale-levels-for-htrts-there-are-no-jump-thresholds-for-wind-direction-observations)), while it operates solely at the raw data scale for LTRTs ([Table 8](#table-8---jump-thresholds-for-the-raw-and-minute-scale-levels-for-ltrts-there-are-no-jump-thresholds-for-wind-direction-observations)). The LTRTs' thresholds are applied in a proportional way till the pre-defined upper limit ([Table 8](#table-8---jump-thresholds-for-the-raw-and-minute-scale-levels-for-ltrts-there-are-no-jump-thresholds-for-wind-direction-observations)). For instance, if the temperature jump threshold for a 1-min interval is 3°C, it will be 9°C for a 3-min interval, but will never exceed the upper limit of 15°C. The upper limits are recommended by the [European Commission](https://publications.jrc.ec.europa.eu/repository/handle/JRC128152) for consecutive 1-hour averaged values. We highlight the lack of RH upper limit prompting us to make an arbitrary yet logical decision to set the value at 80%. In case of a faulty jump detection, an additional check is conducted to label the subsequent value as faulty if it equals the previously identified faulty value.

To overcome the challenge of distinguishing a faulty value from two consecutive ones, we require a minimum of 67% of the data within the past 10-minute window to calculate the median for HTRTs (60-min window for LTRTs). Subsequently, we compare both the two consecutive values with the calculated median, eventually categorising the one with the largest absolute difference from the median as faulty. The same process applies on both raw and (inter-minute) scale levels. Note that if the median cannot be calculated, then no decision can be taken and the datum is annotated as not available.

##### Table 7 - Jump thresholds for the raw and minute scale levels for HTRTs. There are no jump thresholds for wind direction observations.

| Jump Thresholds                   | Temperature | RH   | Wind Spd | Wind Dir | Pressure | Illuminance |
|-----------------------------------|-------------|------|----------|----------|----------|-------------|
| Raw scale level jump threshold    | ≤2°C        | ≤5%  | ≤20m/s   | -        | ≤0.5mb   | ≤97600Lux   |
| Minute scale level jump threshold | ≤3°C        | ≤10% | ≤10m/s   | -        | ≤0.5mb   | ≤97600Lux   |

<br>

##### Table 8 - Jump thresholds for the raw and minute scale levels for LTRTs. There are no jump thresholds for wind direction observations.

| Jump Thresholds                         | Temperature | RH   | Wind Speed | Wind Dir | Pressure | Illuminance |
|-----------------------------------------|-------------|------|------------|----------|----------|-------------|
| Inter-Minute scale level jump threshold | ≤3°C        | ≤10% | ≤10m/s     | -        | ≤0.5mb   | ≤97600Lux   |
| Upper Limit                             | 15°C        | 80%  | 15m/s      | -        | 15mb     | 146400Lux   |

#### d. Interpreting the Annotations

At both of the raw and (inter-) minute scale levels various text annotations are assigned to every single value for each of the 7 investigated meteorological variables ([Table 9](#table-9---all-the-possible-annotations-derived-from-qod-v1-and-reach-the-hourlydaily-level)). At the hourly scale level, all the unique text annotations produced in the previous levels are gathered to provide information about all the detected faults within the hourly slot. 

##### Table 9 - All the possible annotations derived from QoD v1 and reach the hourly/daily level.

|   | Annotation Code                 | Derivation Level | Applies to                                      | Process                    | Description |
|---|---------------------------------|------------------|-------------------------------------------------|----------------------------|-------------|
| 1 | OBC                             | Raw              | All parameters                                  | OBC                        | Value is out of manufacturer's bounds |
| 2 | SPIKE_INST                      | Raw              | All except precipitation and wind direction     | Suspicious Jump Detection  | Value represents a meteorologically unreasonable spike/dip |
| 3 | UNIDENTIFIED SPIKE              | Raw              | All except precipitation and wind direction     | Suspicious Jump Detection  | There is a group of consecutive values with a large difference between them, but there are not enough data to identify which value is the real spike |
| 4 | NO_DATA                         | Raw              | All parameters                                  | Unavailable Data Detection | No data received in the expected timeslot |
| 5 | SHORT_CONST                     | Raw              | All except precipitation                        | Constancy Checks           | Parameter remains constant for longer than expected |
| 6 | LONG_CONST                      | Raw              | All except humidity, illuminance, precipitation | Constancy Checks           | Parameter remains constant for longer than expected |
| 7 | FROZEN_SENSOR                   | Raw              | Wind speed and wind direction                   | Constancy Checks           | Parameter remains constant for longer than expected due to low temperatures |
| 8 | ANOMALOUS_INCREASE              | Minute           | All except precipitation and wind direction     | Suspicious Jump Detection  | Value represents a meteorologically unreasonable spike/dip |
| 9 | UNIDENTIFIED_ANOMALOUS_INCREASE | Minute           | All except precipitation and wind direction     | Suspicious Jump Detection  | There is a group of consecutive values with a large difference between them, but there are not enough data to identify which value is the real spike |