# WeatherXM QoD

## Intro

WeatherXM's QoD (Quality-of-Data) is the algorithm that evaluates the quality of weather data provided by the network's weather stations. The calculated QoD score indicates the confidence level in the quality of the weather data received from a station. We need meaningful and usable data and the QoD score is an attempt to quantify this metric. The end goal is to encourage weather station owners to do their best to comply with the Network's guidelines in order to consistently achieve the best QoD score possible.

QoD involves a series of techniques and processes designed to help us distinguish between expected and unexpected data behaviors. This mechanism includes the following 3 control points and their respective checks:

1. Self Check
   - Out-of-Bounds Check (OBC) (since QoD v1.0)
   - Self Quality Check (SQC) (since QoD v1.0)

2. Reference Check
   - Comparative Quality Control (CQC) (coming in QoD v1.3)

3. Deployment status
   - Indoor Station Detector (ISD) (since Qod v1.1)
   - Solar Obstacle Detector (SOD) (coming in QoD v1.2)
   - Wind Obstacle Detector (WOD) (TBD)

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

### Run the Bacalhau job

```bash
bacalhau docker run \
  --memory 32GB \
  --cpu 12 \
  -i src=ipfs://bafybeifyaa3hnme72i5zrkw2xvheoucturog7i56moztm5bvxie5xhy2ni,dst=/input_2023_12_13 \
  -i src=ipfs://bafybeifd4m73uznnvglugkrqll7xlkqncbvxrfhq6bkuezdplwtw4kzaw4,dst=/input_2023_12_14 \
  ghcr.io/weatherxm-network/qod:0.0.1 -- \
  obc_sqc.iface.file_model_inference \
  --device_id 8bf5b8f0-50db-11ed-960b-b351f0b0cc44 \
  --date "2023-12-14" \
  --day1 /input_2023_12_13/w_tst1/data/1702936551948979.parquet \
  --day2 /input_2023_12_14/w_tst1/data/1702936591988229.parquet \
  --output_file_path /outputs/result
```

You can see an example out put by fetching the results:
```bash
bacalhau get bf059011-e744-40a0-9145-137d6e0803e4
```
