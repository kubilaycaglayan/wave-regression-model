"""Cap working-photo resolution for full-frame Step 1 image operations."""
from __future__ import annotations

import time

WORKING_MAX_DIMENSION = 2048


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
    """Run the baseline route on a capped working image after Step 1 inference."""
    import cv2
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with Timer(timings, "preprocessing.decode_oriented_rgb"):
        image = server.decode_photo(photo)
    try:
        if isinstance(image, (server.Image.Image, server.NativeRGBFrame)):
            original_width, original_height = image.width, image.height
            source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image)
        else:
            original_height, original_width = image.shape[:2]
            source = image

        tensor = _input_tensor(image, 448, server, timings, "step1.tensor")
        with Timer(timings, "preprocessing.step1.segmentation_inference"):
            low_mask = server.STEP1_SESSION.run(["water_mask"], {"pixel_values": tensor})[0][0]

        working_scale = min(1.0, WORKING_MAX_DIMENSION / max(original_width, original_height))
        width = max(1, round(original_width * working_scale))
        height = max(1, round(original_height * working_scale))
        if working_scale < 1.0:
            with Timer(timings, "preprocessing.step1.downsample_working_photo"):
                working_source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        else:
            working_source = source
        with Timer(timings, "preprocessing.step1.expand_mask_to_working_resolution"):
            mask = server.expand_mask(low_mask, width, height)

        radius = max(1, min(12, round(min(width, height) * 0.005)))
        with Timer(timings, "preprocessing.step1.erode_mask"):
            eroded = cv2.erode(
                mask,
                np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                borderType=cv2.BORDER_REPLICATE,
            )
        with Timer(timings, "preprocessing.step1.bounding_box"):
            left, top, box_width, box_height = cv2.boundingRect(eroded)
        with Timer(timings, "preprocessing.step1.water_pixel_count_and_validation"):
            water_pixels = cv2.countNonZero(eroded)
            minimum = max(32, round(width * height * 0.0005))
            if box_width == 0 or box_height == 0 or water_pixels < minimum:
                raise ValueError("Step 1 could not reliably detect water")
        with Timer(timings, "preprocessing.step1.roi_views"):
            roi_pixels = working_source[top:top + box_height, left:left + box_width]
            mask_crop = eroded[top:top + box_height, left:left + box_width]
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
            "image": {"width": original_width, "height": original_height, "format": photo.suffix.lower()},
            "intermediate": {
                "original_image_size_wh": [original_width, original_height],
                "working_image_size_wh": [width, height],
                "step1_water_bbox_xywh": [left, top, box_width, box_height],
                "step2_selected_crop_xywh": list(selected[0]),
                "regressor_tensor_shape": list(model_tensor.shape),
                "regressor_tensor_dtype": str(model_tensor.dtype),
            },
        }
    finally:
        if isinstance(image, server.NativeRGBFrame):
            image.close()
