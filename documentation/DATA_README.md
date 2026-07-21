# Dataset README

## Scope

The release contains hourly station-scale records for 98 stations in Hunan
Province during 2021--2023. Each row is identified by `time + id` and combines
the station precipitation target, GPM IMERG precipitation, near-surface
meteorological variables, geography/terrain variables, land-cover indicators,
ERA5 pressure-level relative vorticity, and jet-informed features.

Time is **Beijing Time (UTC+8)**. The timestamp is interpreted as the ending
time of the preceding hourly station accumulation interval.

## Files

| File | Contents |
|---|---|
| `station_hourly_all_2021_2023.csv` | Complete public table; 206,671 rows and 41 columns. |
| `station_hourly_training_set.csv` | Preserved training split; 165,336 rows. |
| `station_hourly_test_set.csv` | Preserved test split; 41,335 rows. |
| `release_processing_qc.json` | Existing release-processing record, including duplicate removal. |
| `quality_control_report.json` | Independent structural and cross-file checks made during packaging. |

The complete table is exactly partitioned by the supplied training and test
files using the `time + id` key. There is no key overlap between the two
splits. 

## Coordinates

- Each released coordinate is approximately 1--2 km from its source coordinate
  and is rounded to four decimal places.
- `lon_coding = (lon - 108.0) / 7.0` and
  `lat_coding = (lat - 24.0) / 7.0`, calculated from the released coordinates.

## Values and scaling

Several Tianqing/station variables are stored in the preprocessed normalized
scale used by the supplied legacy workflow. `PRE_1h` is the station-observed
hourly precipitation amount in millimetres (mm) and is not normalized in the
released CSV files. `gpm` is the physical GPM precipitation value and `GPM` is
its model-input normalized counterpart. `VO_850` and `VO_500` are physical
relative-vorticity values in s^-1. The table does not include ERA5-Land
precipitation; the abandoned `era5_land_tp_mm` feature is excluded.

The release preserves the existing values; packaging did not re-normalize or
recompute model inputs. For new experiments, estimate any new normalization
parameters on the training set only, then apply the same parameters to the test
set. Do not fit a scaler independently on the test set or on the combined
training-and-test table.

## Missing values

The complete table contains 13 missing values in `gpm`; no other missing fields
were detected by the packaging quality check. Users should document the chosen
handling of these 13 records. The companion normalized `GPM` field should not
be assumed to restore a missing physical `gpm` value without checking the
original processing convention.

## Land cover

The eight `LandOver_*` columns are one-hot indicators. Exactly one is active in
every released row. The codes follow the GLC_FCS30/ESA-LCCS family of land-cover
classification codes used by the source workflow. See `DATA_DICTIONARY.csv` for
the class labels used in this release.
