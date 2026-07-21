# Core code README

Only contains the core PAJE-INet code of the submitting author.

## Core files

1. `paje_net_training.py`
   - Main PAJE-Net training, validation, checkpointing, and prediction script.
   - Defines the Bernoulli--Gamma loss used by the available implementation.
   - Uses the supplied training and test CSV files from `data/`.

2. `jet_feature_generation.py`
   - Converts time-matched jet-axis and wind-field images into station-scale
     directional features, multiscale EJFI values, and four jet embeddings.
   - Expects original station coordinates and source images in a local
     `private_inputs/` folder. Those restricted inputs are not distributed.

3. `jet_patch_encoder.py`
   - Defines `MicroJetEncoder`, the compact CNN used to encode 9 x 9 local
     wind-field patches into the four `Jet_Emb_*` components.

## Running the training script

From the `code` directory:

```powershell
python -m pip install -r requirements.txt
python paje_net_training.py
```

The script reads:

- `../data/station_hourly_training_set.csv`
- `../data/station_hourly_test_set.csv`

Checkpoints and prediction results are written to `code/outputs/`.

The available implementation requires a CUDA-capable environment because the
training script retains `device = 'cuda'` from the research code.

## External CPS-RAUnet++ method

CPS-RAUnet++ is a separately published jet-axis detection method. Its source
code is not redistributed here. The package includes only the present study's
downstream jet-feature construction and PAJE-Net training code.

Gan, J.; Cai, K.; Fan, C.; Deng, X.; Hu, W.; Li, Z.; Wei, P.; Liao, T.; Zhang,
F. *CPS-RAUnet++: A Jet Axis Detection Method Based on Cross-Pseudo Supervision
and Extended Unet++ Model.* Electronics **2025**, 14, 441.
https://doi.org/10.3390/electronics14030441
