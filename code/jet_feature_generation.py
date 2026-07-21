from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import torch
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from jet_patch_encoder import MicroJetEncoder, extract_patch


# ==========================================
# 0. Global configuration and model loading
# ==========================================
IMG_CONFIG = {
    'width': 512,
    'height': 320,
    'lon_range': (0, 160),
    'lat_range': (12, 80)
}

code_dir = Path(__file__).resolve().parent
private_input_dir = code_dir / "private_inputs"
PATHS = {
    'csv_path': private_input_dir / "station_table_with_original_coordinates.csv",
    'img_root': private_input_dir / "era5_wind_images",
    'res_root': private_input_dir / "jet_axis_masks",
    'output_path': code_dir / "outputs" / "station_table_with_jet_features.csv"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
micro_encoder = MicroJetEncoder(in_channels=3, latent_dim=4).to(device)
micro_encoder.eval()


# ==========================================
# 1. Helper functions
# ==========================================
def parse_filename(time_str):
    """
    Convert a timestamp to the image identifier.

    Input example: 2022/2/17 21:00 (zero padding is optional).
    Output example: 22021721.
    """
    try:
        # Parse the primary timestamp format.
        dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
        return dt.strftime("%y%m%d%H")
    except ValueError:
        # Try a general pandas parser as a fallback.
        try:
            dt = pd.to_datetime(time_str)
            return dt.strftime("%y%m%d%H")
        except Exception:
            return None


def geo_to_pixel(lon, lat, config):
    """Convert longitude and latitude to image-pixel coordinates."""
    lon_min, lon_max = config['lon_range']
    lat_min, lat_max = config['lat_range']
    w, h = config['width'], config['height']

    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return x, y


def load_image_pair(file_id):
    """Load an input image and label image and calculate the distance map."""
    input_path = PATHS['img_root'] / f"{file_id}.png"
    label_path = PATHS['res_root'] / f"{file_id}_result.png"

    if not input_path.exists() or not label_path.exists():
        return None

    # 1. Input image.
    img_input = cv2.imread(str(input_path))  # BGR
    if img_input is None:
        return None

    # 2. Label image and distance map.
    img_label = cv2.imread(str(label_path), 0)
    if img_label is None:
        return None

    _, binary_mask = cv2.threshold(img_label, 127, 1, cv2.THRESH_BINARY)
    # The distance transform is calculated once for each image.
    dist_map = distance_transform_edt(1 - binary_mask)

    return img_input, dist_map


# ==========================================
# 2. Core processing workflow
# ==========================================
def main():
    print(f"Reading data: {PATHS['csv_path']}...")
    df = pd.read_csv(PATHS['csv_path'])

    # Define result columns in advance.
    new_columns = ['feat_sin', 'feat_cos', 'ejfi_3', 'ejfi_5', 'ejfi_8',
                   'Jet_Emb_1', 'Jet_Emb_2', 'Jet_Emb_3', 'Jet_Emb_4']

    # Store results by the original DataFrame index. This is faster than
    # repeatedly assigning values to individual DataFrame rows.
    results_map = {}

    print("Processing records grouped by time to reduce image I/O...")

    # Each image is loaded once for all stations at the same time.
    grouped = df.groupby('time')

    for time_str, group_indices in tqdm(grouped.indices.items(), total=len(grouped)):

        # 1. Obtain the image identifier.
        file_id = parse_filename(time_str)
        if file_id is None:
            print(f"Warning: failed to parse timestamp {time_str}")
            continue

        # 2. Load image data once for this time group.
        img_data = load_image_pair(file_id)
        if img_data is None:
            # Missing images leave the corresponding derived fields as NaN.
            continue

        img_input, dist_map = img_data

        # Normalize the image for the CNN.
        img_normalized = img_input.astype(np.float32) / 255.0

        # Separate channels for physical-feature calculations.
        channel_sin = img_input[:, :, 0]
        channel_cos = img_input[:, :, 1]
        channel_spd = img_input[:, :, 2]

        # 3. Process all stations at this timestamp.
        for idx in group_indices:
            row = df.loc[idx]
            lon, lat = row['lon'], row['lat']

            px, py = geo_to_pixel(lon, lat, IMG_CONFIG)

            # Physical feature extraction.
            raw_spd = channel_spd[py, px]
            raw_sin = channel_sin[py, px]
            raw_cos = channel_cos[py, px]
            dist_val = dist_map[py, px]

            # Recover wind speed.
            real_wind_speed = float(raw_spd) / 6.375 if raw_spd > 0 else 0.0

            # Directional sine and cosine.
            if raw_spd == 0:
                f_sin, f_cos = 0.0, 0.0
            else:
                f_sin = (float(raw_sin) / 127.5) - 1.0
                f_cos = (float(raw_cos) / 127.5) - 1.0

            # Calculate EJFI at three bandwidths.
            sigmas = [3, 5, 8]
            ejfi_vals = {}
            for sigma in sigmas:
                influence = np.exp(-(dist_val ** 2) / (2 * sigma ** 2))
                if influence < 1e-4:
                    influence = 0.0
                ejfi_vals[f'ejfi_{sigma}'] = real_wind_speed * influence

            # CNN feature extraction.
            patch = extract_patch(img_normalized, px, py, patch_size=9)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                # Shape: (1, 4) -> one-dimensional NumPy array.
                embedding = micro_encoder(patch_tensor).cpu().numpy().flatten()

            results_map[idx] = {
                'feat_sin': f_sin,
                'feat_cos': f_cos,
                'ejfi_3': ejfi_vals['ejfi_3'],
                'ejfi_5': ejfi_vals['ejfi_5'],
                'ejfi_8': ejfi_vals['ejfi_8'],
                'Jet_Emb_1': embedding[0],
                'Jet_Emb_2': embedding[1],
                'Jet_Emb_3': embedding[2],
                'Jet_Emb_4': embedding[3]
            }

    print("Processing complete. Merging derived features...")

    # Convert the result dictionary to a DataFrame aligned by original index.
    result_df = pd.DataFrame.from_dict(results_map, orient='index')

    # Join the new columns to the source table.
    df_final = df.join(result_df)

    print("Saving output...")
    PATHS['output_path'].parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PATHS['output_path'], index=False)
    print(f"Saved to: {PATHS['output_path']}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        print("The program terminated with an error.")
