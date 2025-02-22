import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew, kurtosis


class AdaptiveDistributionLoss(nn.Module):
    def __init__(self):
        super(AdaptiveDistributionLoss, self).__init__()
        self.total_area = []
        self.background_area = []

    def calculate_areas(self, batch_masks):
        """Calculate foreground and background areas for masks."""
        self.total_area = []
        self.background_area = []

        for mask in batch_masks:
            # Mask is a tensor with shape (1, H, W)
            mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

            cnts, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area_foreground = sum(cv2.contourArea(cnt) for cnt in cnts)
            self.total_area.append(area_foreground)

            h, w = mask_np.shape
            background = (h * w) - area_foreground
            self.background_area.append(background)

    def forward(self, logits, y_true):
        """
        Compute the adaptive loss based on the statistical properties
        of the foreground areas in the dataset.
        """
        y_pred = torch.sigmoid(logits)  # Convert logits to probabilities

        total_area = np.array(self.total_area)
        background_area = np.array(self.background_area)

        # Calculate statistical measures
        skewness = skew(total_area)
        kurtosis_value = kurtosis(total_area, fisher=True)

        # Apply transformations based on skewness and kurtosis
        if skewness < 0:  # Negative skewness
            if skewness <= -1:
                # Fisher's transformation
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * self._fisher_transformation(y_pred)
                    + (1 - y_true) * self._fisher_transformation(1 - y_pred)
                )
            elif -1 < skewness <= -0.5:
                # Logit transformation
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * self._logit(y_pred)
                    + (1 - y_true) * self._logit(1 - y_pred)
                )
            else:
                # Arcsine transformation
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * torch.asin(torch.sqrt(y_pred))
                    + (1 - y_true) * torch.asin(torch.sqrt(1 - y_pred))
                )
        else:  # Positive skewness
            if skewness >= 1:
                # Log base 10 transformation
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * torch.log10(y_pred)
                    + (1 - y_true) * torch.log10(1 - y_pred)
                )
            elif 0.5 <= skewness < 1:
                # Natural log transformation
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
                )
            elif 0 < skewness <= 0.5 and kurtosis_value < 0:
                # Log base 10 for near-symmetrical skewness and platykurtic kurtosis
                y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
                loss = -torch.mean(
                    y_true * torch.log10(y_pred)
                    + (1 - y_true) * torch.log10(1 - y_pred)
                )
            else:
                # Binary Cross-Entropy with Dice Loss
                loss = self._bce_dice_loss(y_true, y_pred)

        return loss

    @staticmethod
    def _fisher_transformation(p):
        """Apply Fisher transformation."""
        return 0.5 * torch.log((1 + p) / (1 - p))

    @staticmethod
    def _logit(p):
        """Apply logit transformation."""
        return torch.log(p / (1 - p))

    @staticmethod
    def _bce_dice_loss(y_true, y_pred):
        """Binary Cross-Entropy combined with Dice Loss."""
        bce = F.binary_cross_entropy(y_pred, y_true)
        smooth = 1e-7
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return bce + (1 - dice)
