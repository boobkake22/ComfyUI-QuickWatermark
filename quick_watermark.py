from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ALIGNMENTS = [
    "bottom-right",
    "center",
    "top-center",
    "top-left",
    "top-right",
    "center-right",
    "center-left",
    "bottom-center",
    "bottom-left",
]

WATERMARK_PRESETS = {
    "speaker-white": "Speaker_Icon_White.png",
    "speaker-black": "Speaker_Icon_Black.png",
}


class QuickWatermarkVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_preset": (list(WATERMARK_PRESETS.keys()),),
                "alignment": (ALIGNMENTS,),
                "offset_percentage": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "resize_percentage": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.1, "max": 200.0, "step": 0.1},
                ),
                "opacity": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
            },
            "optional": {
                "watermark": ("IMAGE",),
                "watermark_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_watermark"
    CATEGORY = "image/watermark"
    SEARCH_ALIASES = ["quick watermark", "video watermark", "png watermark"]

    def apply_watermark(
        self,
        image,
        watermark_preset,
        alignment,
        offset_percentage,
        resize_percentage,
        opacity,
        watermark=None,
        watermark_mask=None,
    ):
        self._validate_image_tensor(image, "image")
        frame_count, frame_height, frame_width, frame_channels = image.shape
        if frame_channels < 3:
            raise ValueError("image must have at least 3 channels")

        opacity_value = self._clamp_percentage(opacity)
        if opacity_value <= 0.0:
            return (image.clone(),)

        watermark_rgb, alpha = self._resolve_watermark(
            image,
            watermark_preset,
            watermark,
            watermark_mask,
        )
        self._validate_watermark_batch(watermark_rgb, alpha, frame_count)

        target_width = max(1, int(round(frame_width * resize_percentage / 100.0)))
        target_height = max(
            1,
            int(round(watermark_rgb.shape[1] * target_width / watermark_rgb.shape[2])),
        )

        watermark_rgb = self._resize_channel_last(watermark_rgb, target_height, target_width)
        alpha = self._resize_channel_last(alpha, target_height, target_width).clamp(0.0, 1.0)
        alpha = alpha * opacity_value

        offset_x = int(round(frame_width * offset_percentage / 100.0))
        offset_y = int(round(frame_height * offset_percentage / 100.0))
        x, y = self._placement(
            alignment,
            frame_width,
            frame_height,
            target_width,
            target_height,
            offset_x,
            offset_y,
        )

        output = image.clone()
        for frame_index in range(frame_count):
            watermark_index = 0 if watermark_rgb.shape[0] == 1 else frame_index
            alpha_index = 0 if alpha.shape[0] == 1 else frame_index
            self._composite_into_frame(
                output,
                frame_index,
                watermark_rgb[watermark_index],
                alpha[alpha_index],
                x,
                y,
            )

        return (output.clamp(0.0, 1.0),)

    @staticmethod
    def _validate_image_tensor(value, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape [B,H,W,C]")
        if value.shape[1] <= 0 or value.shape[2] <= 0 or value.shape[3] <= 0:
            raise ValueError(f"{name} must have non-empty height, width, and channels")

    def _resolve_watermark(self, image, watermark_preset, watermark, watermark_mask):
        if watermark is not None:
            self._validate_image_tensor(watermark, "watermark")
            if watermark.shape[-1] < 3:
                raise ValueError("watermark must have at least 3 channels")
            watermark_rgb = watermark[..., :3].to(device=image.device, dtype=image.dtype)
            alpha = self._watermark_alpha(watermark, watermark_mask, image)
            return watermark_rgb, alpha

        return self._load_preset_watermark(watermark_preset, image)

    @staticmethod
    def _load_preset_watermark(watermark_preset, image):
        if watermark_preset not in WATERMARK_PRESETS:
            raise ValueError(f"Unknown watermark preset: {watermark_preset}")

        asset_path = Path(__file__).with_name("assets") / WATERMARK_PRESETS[watermark_preset]
        if not asset_path.exists():
            raise FileNotFoundError(f"Watermark preset file was not found: {asset_path}")

        with Image.open(asset_path) as preset_image:
            rgba = preset_image.convert("RGBA")
            preset = np.array(rgba, dtype=np.float32) / 255.0

        preset = torch.from_numpy(preset).unsqueeze(0).to(
            device=image.device,
            dtype=image.dtype,
        )
        return preset[..., :3], preset[..., 3:4]

    @staticmethod
    def _validate_watermark_batch(watermark_rgb, alpha, frame_count):
        watermark_count = watermark_rgb.shape[0]
        if watermark_count not in (1, frame_count):
            raise ValueError(
                "watermark batch must contain either 1 image or the same number "
                "of images as the input image batch"
            )

        if alpha.shape[0] not in (1, watermark_count, frame_count):
            raise ValueError(
                "watermark alpha batch must contain either 1 alpha image, the same "
                "number as the watermark batch, or the same number as the input image batch"
            )

    @staticmethod
    def _clamp_percentage(value):
        return max(0.0, min(1.0, float(value) / 100.0))

    def _watermark_alpha(self, watermark, watermark_mask, image):
        if watermark_mask is not None:
            alpha = 1.0 - self._normalize_mask(watermark_mask, image)
        elif watermark.shape[-1] >= 4:
            alpha = watermark[..., 3:4].to(device=image.device, dtype=image.dtype)
        else:
            alpha = torch.ones(
                watermark.shape[0],
                watermark.shape[1],
                watermark.shape[2],
                1,
                device=image.device,
                dtype=image.dtype,
            )

        if alpha.shape[0] not in (1, watermark.shape[0], image.shape[0]):
            raise ValueError(
                "watermark_mask batch must contain either 1 mask, the same number "
                "of masks as the watermark batch, or the same number as the input image batch"
            )

        if alpha.shape[0] == image.shape[0] and watermark.shape[0] == 1:
            return alpha

        if alpha.shape[0] != watermark.shape[0]:
            alpha = alpha[:1]

        return alpha

    @staticmethod
    def _normalize_mask(mask, image):
        if not isinstance(mask, torch.Tensor):
            raise TypeError("watermark_mask must be a torch.Tensor")

        mask = mask.to(device=image.device, dtype=image.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(-1)
        elif mask.ndim == 4:
            if mask.shape[1] == 1 and mask.shape[-1] != 1:
                mask = mask.movedim(1, -1)
            if mask.shape[-1] != 1:
                mask = mask[..., :1]
        else:
            raise ValueError("watermark_mask must have shape [H,W], [B,H,W], or [B,H,W,C]")

        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _resize_channel_last(tensor, height, width):
        channel_first = tensor.movedim(-1, 1)
        resized = F.interpolate(
            channel_first,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.movedim(1, -1)

    @staticmethod
    def _placement(
        alignment,
        frame_width,
        frame_height,
        watermark_width,
        watermark_height,
        offset_x,
        offset_y,
    ):
        center_x = (frame_width - watermark_width) // 2
        center_y = (frame_height - watermark_height) // 2
        right_x = frame_width - watermark_width - offset_x
        bottom_y = frame_height - watermark_height - offset_y

        placements = {
            "center": (center_x, center_y),
            "top-center": (center_x, offset_y),
            "top-left": (offset_x, offset_y),
            "top-right": (right_x, offset_y),
            "center-right": (right_x, center_y),
            "center-left": (offset_x, center_y),
            "bottom-center": (center_x, bottom_y),
            "bottom-right": (right_x, bottom_y),
            "bottom-left": (offset_x, bottom_y),
        }
        return placements[alignment]

    @staticmethod
    def _composite_into_frame(output, frame_index, watermark_rgb, alpha, x, y):
        frame_height, frame_width = output.shape[1], output.shape[2]
        watermark_height, watermark_width = watermark_rgb.shape[0], watermark_rgb.shape[1]

        frame_x0 = max(0, x)
        frame_y0 = max(0, y)
        frame_x1 = min(frame_width, x + watermark_width)
        frame_y1 = min(frame_height, y + watermark_height)

        if frame_x0 >= frame_x1 or frame_y0 >= frame_y1:
            return

        watermark_x0 = frame_x0 - x
        watermark_y0 = frame_y0 - y
        watermark_x1 = watermark_x0 + (frame_x1 - frame_x0)
        watermark_y1 = watermark_y0 + (frame_y1 - frame_y0)

        source = watermark_rgb[watermark_y0:watermark_y1, watermark_x0:watermark_x1, :]
        source_alpha = alpha[watermark_y0:watermark_y1, watermark_x0:watermark_x1, :]
        target = output[frame_index, frame_y0:frame_y1, frame_x0:frame_x1, :3]

        output[frame_index, frame_y0:frame_y1, frame_x0:frame_x1, :3] = (
            source * source_alpha + target * (1.0 - source_alpha)
        )


NODE_CLASS_MAPPINGS = {
    "QuickWatermarkVideo": QuickWatermarkVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickWatermarkVideo": "Quick Watermark Video",
}
