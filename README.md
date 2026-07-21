# Station-Scale Jet-Informed Precipitation Dataset

This repository package accompanies the manuscript **A Station-Scale
Jet-Informed Dataset for GPM IMERG Precipitation Correction over Hunan
Province, China**.

It contains the processed public dataset, fixed training and test splits,
metadata for 98 stations, the three core PAJE-Net source files, a field
dictionary, and quality-control records.

## Contents

### `data/`

- `station_hourly_all_2021_2023.csv`: complete processed station-hour table.
- `station_hourly_training_set.csv`: fixed training split.
- `station_hourly_test_set.csv`: fixed test split.
- `quality_control_report.json`: structural and cross-file validation results.
- `release_processing_qc.json`: release-processing and duplicate-removal record.

### `code/`

- `paje_net_training.py`: PAJE-Net training and prediction.
- `jet_feature_generation.py`: station-scale EJFI and jet-feature construction.
- `jet_patch_encoder.py`: compact CNN jet-patch encoder.
- `requirements.txt`: imported Python dependencies.
- `README.md`: code description and execution notes.

### `documentation/`

- `DATA_README.md`: dataset scope, time convention, coordinate policy, scaling,
  and missing-value notes.
- `DATA_DICTIONARY.csv`: definitions for all 41 columns.
- `SHA256SUMS.txt`: integrity hashes for the deposited files.

## Time and coordinates

Timestamps use Beijing Time (UTC+8). Station latitude and longitude are kept to four decimal places.

## External method

CPS-RAUnet++ source code is not redistributed. The external jet-axis detection
method should be cited as:

Gan, J.; Cai, K.; Fan, C.; Deng, X.; Hu, W.; Li, Z.; Wei, P.; Liao, T.; Zhang,
F. CPS-RAUnet++: A Jet Axis Detection Method Based on Cross-Pseudo Supervision
and Extended Unet++ Model. *Electronics* **2025**, *14*, 441.
https://doi.org/10.3390/electronics14030441
