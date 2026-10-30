"""Faster Step 1 resize · linear interpolation.

Original v001 coordinate-based lower-water crop implementation, modified only by this candidate.

Experimental route: Step 1's water mask selects the source-photo crop. The
selected source pixels and mask are resampled directly into the only RGB photo
canvas used by the regressor. Step 2 segmentation is intentionally omitted.
"""
from __future__ import annotations

import io
import math
import time


class Timer:
    def __init__(self, timings: dict[str, float], key: str):
        self.timings = timings
        self.key = key

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.timings[self.key] = (time.perf_counter() - self.started) * 1000.0


def _input_tensor(image, size: int, server, timings: dict[str, float], prefix: str):
    import cv2
    import numpy as np

    source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image, dtype=np.uint8)
    interpolation = cv2.INTER_LINEAR if size < max(source.shape[:2]) else cv2.INTER_LINEAR_EXACT
    with Timer(timings, f"preprocessing.{prefix}.resize_rgb"):
        resized = cv2.resize(source, (size, size), interpolation=interpolation)
    with Timer(timings, f"preprocessing.{prefix}.convert_to_float32_and_scale"):
        pixels = resized.astype(np.float32)
        pixels *= np.float32(1.0 / 255.0)
    with Timer(timings, f"preprocessing.{prefix}.normalize_mean_std"):
        pixels = (pixels - server.MEAN) / server.STD
    with Timer(timings, f"preprocessing.{prefix}.transpose_and_contiguous_tensor"):
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor


def run_pipeline(photo, server, reference_pipeline):
    """Return the usual benchmark record for the one-image coordinate route."""
    import cv2
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with Timer(timings, "preprocessing.decode_oriented_rgb"):
        image = server.decode_photo(photo)
    try:
        if isinstance(image, (server.NativeRGBFrame, server.Image.Image)):
            width, height = image.width, image.height
        else:
            height, width = image.shape[:2]
        source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image)

        step1_tensor = _input_tensor(image, 448, server, timings, "step1_tensor")
        with Timer(timings, "preprocessing.step1.segmentation_inference"):
            predicted_mask = server.STEP1_SESSION.run(
                ["water_mask"], {"pixel_values": step1_tensor}
            )[0][0]

        with Timer(timings, "preprocessing.step1.mask_resize_to_crop_grid"):
            mask_224 = cv2.resize(
                predicted_mask.astype(np.uint8), (224, 224),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ).astype(bool)
        black_224 = np.zeros((224, 224), dtype=bool)
        with Timer(timings, "preprocessing.step1.lower_water_crop_selection"):
            selected = server.select_crop(mask_224, black_224)
            if selected is None:
                raise ValueError("Step 1 could not reliably select a lower-water crop")

        left, top, right, bottom = selected[0]
        with Timer(timings, "preprocessing.coordinates.map_crop_to_source"):
            x0 = max(0, min(width, math.floor(left * width / 224)))
            x1 = max(x0 + 1, min(width, math.ceil(right * width / 224)))
            y0 = max(0, min(height, math.floor(top * height / 224)))
            y1 = max(y0 + 1, min(height, math.ceil(bottom * height / 224)))
            source_roi = source[y0:y1, x0:x1]
            mask_roi = mask_224[top:bottom, left:right]

        with Timer(timings, "preprocessing.final_canvas.allocate_and_direct_resize"):
            scale = min(224 / source_roi.shape[1], 224 / source_roi.shape[0])
            resized_width = max(1, min(224, round(source_roi.shape[1] * scale)))
            resized_height = max(1, min(224, round(source_roi.shape[0] * scale)))
            offset_x = (224 - resized_width) // 2
            offset_y = (224 - resized_height) // 2
            final_canvas = np.zeros((224, 224, 3), dtype=np.uint8)
            canvas_roi = final_canvas[
                offset_y:offset_y + resized_height,
                offset_x:offset_x + resized_width,
            ]
            cv2.resize(
                source_roi, (resized_width, resized_height), dst=canvas_roi,
                interpolation=cv2.INTER_LANCZOS4,
            )

        with Timer(timings, "preprocessing.final_mask.resize_and_apply"):
            final_mask = cv2.resize(
                mask_roi.astype(np.uint8), (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            canvas_roi[final_mask == 0] = 0

        with Timer(timings, "preprocessing.final_image.pil_materialization"):
            final_image = server.Image.fromarray(final_canvas, mode="RGB")
        with Timer(timings, "preprocessing.final_image.jpeg_encode"):
            encoded = io.BytesIO()
            final_image.save(encoded, format="JPEG", quality=95, subsampling=0)
            final_jpeg = encoded.getvalue()
        with Timer(timings, "preprocessing.final_image.jpeg_decode_and_rgb_convert"):
            with server.Image.open(io.BytesIO(final_jpeg)) as decoded:
                decoded.load()
                final_rgb = decoded.convert("RGB")
        with Timer(timings, "preprocessing.final_tensor.rgb_to_float32_and_scale"):
            model_pixels = np.asarray(final_rgb, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        with Timer(timings, "preprocessing.final_tensor.normalize_mean_std"):
            normalized = (model_pixels - server.MEAN) / server.STD
        with Timer(timings, "preprocessing.final_tensor.transpose_and_contiguous_tensor"):
            model_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocessing_total = (time.perf_counter() - started) * 1000.0

        inference_started = time.perf_counter()
        with Timer(timings, "waviness_inference.v18_regressor_onnxruntime"):
            regressor_output = server.REGRESSOR_SESSION.run(
                ["scores"], {"images": model_tensor}
            )[0]
        with Timer(timings, "waviness_inference.score_extraction"):
            score = float(regressor_output.reshape(-1)[0])
        inference_total = (time.perf_counter() - inference_started) * 1000.0

        return {
            "prediction": score,
            "preprocessing_total_ms": preprocessing_total,
            "waviness_inference_total_ms": inference_total,
            "total_raw_to_score_ms": (time.perf_counter() - started) * 1000.0,
            "timings_ms": timings,
            "image": {"width": width, "height": height, "format": photo.suffix.lower()},
            "intermediate": {
                "step1_mask_crop_xyxy_224": [left, top, right, bottom],
                "source_crop_xyxy": [x0, y0, x1, y1],
                "regressor_tensor_shape": list(model_tensor.shape),
                "regressor_tensor_dtype": str(model_tensor.dtype),
            },
        }
    finally:
        if isinstance(image, server.NativeRGBFrame):
            image.close()
