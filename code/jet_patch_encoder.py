import torch.nn as nn
import numpy as np
import cv2


# ==========================================
# 1. Compact CNN encoder for 9 x 9 patches
# ==========================================
class MicroJetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super(MicroJetEncoder, self).__init__()

        # Architecture:
        # Input 9 x 9 -> Conv(3 x 3) -> 7 x 7
        # 7 x 7 -> Conv(3 x 3) -> 5 x 5
        # Flatten 5 x 5 -> fully connected layer

        self.features = nn.Sequential(
            # First layer: basic gradients (9 x 9 -> 7 x 7)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Second layer: higher-order texture/curvature (7 x 7 -> 5 x 5)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Fully connected input dimension: 32 channels x 5 x 5 = 800.
        self.fc = nn.Linear(32 * 5 * 5, latent_dim)

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x shape: (batch, 3, 9, 9)
        x = self.features(x)  # -> (batch, 32, 5, 5)
        x = x.view(x.size(0), -1)  # -> (batch, 800)
        embedding = self.fc(x)  # -> (batch, latent_dim)
        return embedding


# ==========================================
# 2. Extract a centered image patch
# ==========================================
def extract_patch(image, center_x, center_y, patch_size=9):
    """Extract a patch centered at (center_x, center_y)."""
    h, w, c = image.shape
    half_size = patch_size // 2

    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size + 1  # +1 preserves odd patch_size length.
    y2 = center_y + half_size + 1

    # Determine boundary padding.
    pad_top = abs(min(0, y1))
    pad_bottom = max(0, y2 - h)
    pad_left = abs(min(0, x1))
    pad_right = max(0, x2 - w)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        # Replicate edge pixels to preserve field continuity.
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    patch = image[y1:y2, x1:x2, :]

    # Final safeguard to guarantee the requested patch dimensions.
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))

    return patch
