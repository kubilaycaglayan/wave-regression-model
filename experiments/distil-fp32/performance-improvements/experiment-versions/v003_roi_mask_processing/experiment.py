"""Optimize Step 1 mask expansion by limiting full-resolution work to its ROI."""
from __future__ import annotations

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
    interpolation = cv2.INTER_AREA if size < max(source.shape[:2]) else cv2.INTER_LINEAR_EXACT
    with Timer(timings, f"preprocessing.{prefix}.resize_rgb"):
        resized = cv2.resize(source, (size, size), interpolation=interpolation)
    with Timer(timings, f"preprocessing.{prefix}.uint8_to_float32_and_scale"):
        pixels = resized.astype(np.float32)
        pixels *= np.float32(1.0 / 255.0)
    with Timer(timings, f"preprocessing.{prefix}.normalize_mean_std"):
        pixels = (pixels - server.MEAN) / server.STD
    with Timer(timings, f"preprocessing.{prefix}.transpose_and_contiguous_tensor"):
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor


def _canvas(image, server, timings, prefix):
    scale = min(224 / image.width, 224 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    with Timer(timings, f"{prefix}.lanczos_resize"):
        resized = server.resize_lanczos(image, size)
    with Timer(timings, f"{prefix}.black_canvas_allocate_and_paste"):
        result = server.Image.new("RGB", (224, 224), (0, 0, 0))
        result.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
    return result


def run_pipeline(photo, server, reference_pipeline):
    """Run the full baseline route with ROI-limited Step 1 mask processing."""
    import cv2
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with Timer(timings, "preprocessing.decode_oriented_rgb"):
        image = server.decode_photo(photo)
    try:
        if isinstance(image, (server.Image.Image, server.NativeRGBFrame)):
            width, height = image.width, image.height
            source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image)
        else:
            height, width = image.shape[:2]
            source = image

        tensor = _input_tensor(image, 448, server, timings, "step1.tensor")
        with Timer(timings, "preprocessing.step1.segmentation_inference"):
            low_mask = server.STEP1_SESSION.run(["water_mask"], {"pixel_values": tensor})[0][0]

        # Approximate the source-space erosion radius on the model mask grid.
        radius = max(1, min(12, round(min(width, height) * 0.005)))
        mh, mw = low_mask.shape
        rx = max(1, math.ceil(radius * mw / width))
        ry = max(1, math.ceil(radius * mh / height))
        with Timer(timings, "preprocessing.step1.native_mask_erode_and_bbox"):
            native_eroded = cv2.erode(
                np.asarray(low_mask, dtype=np.uint8),
                np.ones((2 * ry + 1, 2 * rx + 1), dtype=np.uint8),
                borderType=cv2.BORDER_REPLICATE,
            )
            nx, ny, nw, nh = cv2.boundingRect(native_eroded)
            if nw == 0 or nh == 0:
                raise ValueError("Step 1 could not reliably detect water")

        # Nearest-exact expansion maps the approximate native bbox to a slightly
        # padded source ROI; padding protects boundary pixels from rounding.
        with Timer(timings, "preprocessing.step1.map_native_bbox_to_source_roi"):
            pad_x = math.ceil(width / mw) + radius
            pad_y = math.ceil(height / mh) + radius
            left = max(0, math.floor(nx * width / mw) - pad_x)
            top = max(0, math.floor(ny * height / mh) - pad_y)
            right = min(width, math.ceil((nx + nw) * width / mw) + pad_x)
            bottom = min(height, math.ceil((ny + nh) * height / mh) + pad_y)
            roi_source = source[top:bottom, left:right]

        # Resize the mask using source-image coordinates so the ROI corresponds
        # to its location, rather than stretching the complete mask into it.
        with Timer(timings, "preprocessing.step1.expand_mask_to_roi"):
            xs = np.clip(((np.arange(left, right) + 0.5) * mw / width).astype(np.int32), 0, mw - 1)
            ys = np.clip(((np.arange(top, bottom) + 0.5) * mh / height).astype(np.int32), 0, mh - 1)
            roi_mask = np.ascontiguousarray(low_mask[ys[:, None], xs[None, :]], dtype=np.uint8)
        with Timer(timings, "preprocessing.step1.erode_roi_mask"):
            eroded_roi = cv2.erode(
                roi_mask,
                np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
                borderType=cv2.BORDER_REPLICATE,
            )
        with Timer(timings, "preprocessing.step1.bounding_box"):
            local_left, local_top, box_width, box_height = cv2.boundingRect(eroded_roi)
            left += local_left
            top += local_top
        with Timer(timings, "preprocessing.step1.water_pixel_count_and_validation"):
            water_pixels = cv2.countNonZero(eroded_roi)
            minimum = max(32, round(width * height * 0.0005))
            if box_width == 0 or box_height == 0 or water_pixels < minimum:
                raise ValueError("Step 1 could not reliably detect water")
        with Timer(timings, "preprocessing.step1.roi_views"):
            roi_pixels = roi_source[local_top:local_top + box_height, local_left:local_left + box_width]
            mask_crop = eroded_roi[local_top:local_top + box_height, local_left:local_left + box_width]
        with Timer(timings, "preprocessing.step1.masked_roi_copy"):
            masked_pixels = cv2.bitwise_and(roi_pixels, roi_pixels, mask=mask_crop)
        with Timer(timings, "preprocessing.step1.pil_image_from_masked_roi"):
            masked = server.Image.fromarray(masked_pixels, mode="RGB")

        step1_image = _canvas(masked, server, timings, "preprocessing.step1.canvas")
        with Timer(timings, "preprocessing.step1.jpeg_encode"):
            buffer = server.io.BytesIO()
            step1_image.save(buffer, format="JPEG", quality=95)
            jpeg = buffer.getvalue()
        with Timer(timings, "preprocessing.step1.jpeg_decode_and_rgb_convert"):
            with server.Image.open(server.io.BytesIO(jpeg)) as decoded:
                decoded.load()
                step1_rgb = decoded.convert("RGB")

        with Timer(timings, "preprocessing.step2.black_area_mask"):
            rgb = np.asarray(step1_rgb, dtype=np.uint8)
            black = np.all(rgb <= 12, axis=2)
        step2_tensor = _input_tensor(step1_rgb, 384, server, timings, "step2.tensor")
        with Timer(timings, "preprocessing.step2.segmentation_inference"):
            low_water = server.STEP2_SESSION.run(["water_mask"], {"pixel_values": step2_tensor})[0][0]
        with Timer(timings, "preprocessing.step2.expand_mask_to_canvas"):
            expanded_water = server.expand_mask(low_water, 224, 224)
        with Timer(timings, "preprocessing.step2.mask_convert_and_exclude_black"):
            water = expanded_water.astype(bool) & ~black
        with Timer(timings, "preprocessing.step2.crop_selection"):
            selected = server.select_crop(water, black)
            if selected is None:
                raise ValueError("Step 2 could not reliably select a lower-water crop")
        with Timer(timings, "preprocessing.step2.crop_extract"):
            selected_image = step1_rgb.crop(selected[0])
        final_image = _canvas(selected_image, server, timings, "preprocessing.step2.canvas")

        with Timer(timings, "preprocessing.final.jpeg_encode"):
            encoded = server.io.BytesIO()
            final_image.save(encoded, format="JPEG", quality=95, subsampling=0)
            final_jpeg = encoded.getvalue()
        with Timer(timings, "preprocessing.final.jpeg_decode_and_rgb_convert"):
            with server.Image.open(server.io.BytesIO(final_jpeg)) as decoded:
                decoded.load()
                final_rgb = decoded.convert("RGB")
        with Timer(timings, "preprocessing.final.rgb_to_float32_scale"):
            pixels = np.asarray(final_rgb, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        with Timer(timings, "preprocessing.final.normalize_mean_std"):
            normalized = (pixels - server.MEAN) / server.STD
        with Timer(timings, "preprocessing.final.transpose_and_contiguous_tensor"):
            model_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocessing_total = (time.perf_counter() - started) * 1000.0

        infer_timings: dict[str, float] = {}
        inference_started = time.perf_counter()
        with Timer(infer_timings, "waviness_inference.v18_regressor_onnxruntime"):
            output = server.REGRESSOR_SESSION.run(["scores"], {"images": model_tensor})[0]
        with Timer(infer_timings, "waviness_inference.score_extraction"):
            score = float(output.reshape(-1)[0])
        inference_total = (time.perf_counter() - inference_started) * 1000.0
        timings.update(infer_timings)
        return {
            "prediction": score,
            "preprocessing_total_ms": preprocessing_total,
            "waviness_inference_total_ms": inference_total,
            "total_raw_to_score_ms": (time.perf_counter() - started) * 1000.0,
            "timings_ms": timings,
            "image": {"width": width, "height": height, "format": photo.suffix.lower()},
            "intermediate": {
                "step1_water_bbox_xywh": [left, top, box_width, box_height],
                "step2_selected_crop_xywh": list(selected[0]),
                "regressor_tensor_shape": list(model_tensor.shape),
                "regressor_tensor_dtype": str(model_tensor.dtype),
            },
        }
    finally:
        if isinstance(image, server.NativeRGBFrame):
            image.close()
