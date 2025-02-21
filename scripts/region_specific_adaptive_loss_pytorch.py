import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew, kurtosis


class RegionSpecificAdaptiveLoss(nn.Module):
    def __init__(self, num_regions=4):
        super(RegionSpecificAdaptiveLoss, self).__init__()
        self.num_regions = num_regions

    def calculate_areas(self, batch_masks):
        total_area = []
        background_area = []

        for mask in batch_masks:
            mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area_foreground = sum(cv2.contourArea(cnt) for cnt in cnts)
            total_area.append(area_foreground)
            h, w = mask_np.shape
            background = (h * w) - area_foreground
            background_area.append(background)

        return total_area, background_area

    def adaptive_distribution_loss(self, logits, y_true, total_area):
        y_pred = torch.sigmoid(logits)
        skewness = skew(total_area)
        kurtosis_value = kurtosis(total_area, fisher=True)

        y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)

        if skewness < 0:
            if skewness <= -1:
                loss = -torch.mean(
                    y_true * self.fisher_transformation(y_pred)
                    + (1 - y_true) * self.fisher_transformation(1 - y_pred)
                )
            elif -1 < skewness <= -0.5:
                loss = -torch.mean(
                    y_true * self.logit(y_pred) + (1 - y_true) * self.logit(1 - y_pred)
                )
            else:
                loss = -torch.mean(
                    y_true * torch.asin(torch.sqrt(y_pred))
                    + (1 - y_true) * torch.asin(torch.sqrt(1 - y_pred))
                )
        else:
            if skewness >= 1:
                loss = -torch.mean(
                    y_true * torch.log10(y_pred)
                    + (1 - y_true) * torch.log10(1 - y_pred)
                )
            elif 0.5 <= skewness < 1:
                loss = -torch.mean(
                    y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
                )
            elif 0 < skewness <= 0.5 and kurtosis_value < 0:
                loss = -torch.mean(
                    y_true * torch.log10(y_pred)
                    + (1 - y_true) * torch.log10(1 - y_pred)
                )
            else:
                loss = self.bce_dice_loss(y_true, y_pred)
        return loss

    @staticmethod
    def fisher_transformation(p):
        return 0.5 * torch.log((1 + p) / (1 - p))

    @staticmethod
    def logit(p):
        return torch.log(p / (1 - p))

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        bce = F.binary_cross_entropy(y_pred, y_true)
        smooth = 1e-7
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return bce + (1 - dice)

    def forward(self, logits, y_true):
        batch_size, _, height, width = y_true.shape
        region_height, region_width = (
            height // self.num_regions,
            width // self.num_regions,
        )

        region_losses = []
        region_weights = []

        for i in range(self.num_regions):
            for j in range(self.num_regions):
                pixel_range = (
                    i * region_height,
                    (i + 1) * region_height,
                    j * region_width,
                    (j + 1) * region_width,
                )

                y_true_region = y_true[
                    :,
                    :,
                    pixel_range[0] : pixel_range[1],
                    pixel_range[2] : pixel_range[3],
                ]
                logits_region = logits[
                    :,
                    :,
                    pixel_range[0] : pixel_range[1],
                    pixel_range[2] : pixel_range[3],
                ]

                total_area, _ = self.calculate_areas(y_true_region)
                region_loss = self.adaptive_distribution_loss(
                    logits_region, y_true_region, total_area
                )

                fp = torch.sum(
                    torch.clamp(
                        torch.round(torch.sigmoid(logits_region)) - y_true_region, min=0
                    )
                )
                fn = torch.sum(
                    torch.clamp(
                        y_true_region - torch.round(torch.sigmoid(logits_region)), min=0
                    )
                )
                region_weight = 1 + fp + fn

                region_losses.append(region_loss)
                region_weights.append(region_weight)

        region_losses = torch.stack(region_losses)
        region_weights = torch.tensor(region_weights, device=y_true.device)
        final_loss = torch.sum(region_losses * region_weights) / torch.sum(
            region_weights
        )

        return final_loss
