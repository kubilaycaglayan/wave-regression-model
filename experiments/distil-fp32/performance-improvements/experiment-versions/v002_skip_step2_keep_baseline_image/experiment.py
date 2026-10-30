"""Skip Step 2 while preserving v000's Step 1 image construction."""
from __future__ import annotations

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


def run_pipeline(photo, server, reference_pipeline):
    """Run baseline preprocessing through Step 1, then infer directly."""
    import cv2
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with Timer(timings, "preprocessing.decode_oriented_rgb"):
        image = server.decode_photo(photo)
    try:
        if isinstance(image, (server.Image.Image, server.NativeRGBFrame)):
            width, height = image.width, image.height
        else:
            height, width = image.shape[:2]

        step1_tensor = _input_tensor(image, 448, server, timings, "step1.tensor")
        with Timer(timings, "preprocessing.step1.segmentation_inference"):
            low_mask = server.STEP1_SESSION.run(
                ["water_mask"], {"pixel_values": step1_tensor}
            )[0][0]
        with Timer(timings, "preprocessing.step1.expand_mask_to_source"):
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
            source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image)
            roi_pixels = source[top:top + box_height, left:left + box_width]
            roi_mask = eroded[top:top + box_height, left:left + box_width]
        with Timer(timings, "preprocessing.step1.masked_roi_copy"):
            masked_pixels = cv2.bitwise_and(roi_pixels, roi_pixels, mask=roi_mask)
        with Timer(timings, "preprocessing.step1.pil_image_from_masked_roi"):
            masked = server.Image.fromarray(masked_pixels, mode="RGB")

        scale = min(224 / masked.width, 224 / masked.height)
        size = (max(1, round(masked.width * scale)), max(1, round(masked.height * scale)))
        with Timer(timings, "preprocessing.step1.canvas.lanczos_resize"):
            resized = server.resize_lanczos(masked, size)
        with Timer(timings, "preprocessing.step1.canvas.black_canvas_allocate_and_paste"):
            step1_image = server.Image.new("RGB", (224, 224), (0, 0, 0))
            step1_image.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))

        with Timer(timings, "preprocessing.step1.jpeg_encode"):
            step1_bytes = server.io.BytesIO()
            step1_image.save(step1_bytes, format="JPEG", quality=95)
            step1_jpeg = step1_bytes.getvalue()
        with Timer(timings, "preprocessing.step1.jpeg_decode_and_rgb_convert"):
            with server.Image.open(server.io.BytesIO(step1_jpeg)) as decoded:
                decoded.load()
                final_rgb = decoded.convert("RGB")

        with Timer(timings, "preprocessing.final.rgb_to_float32_scale"):
            model_pixels = np.asarray(final_rgb, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        with Timer(timings, "preprocessing.final.normalize_mean_std"):
            normalized = (model_pixels - server.MEAN) / server.STD
        with Timer(timings, "preprocessing.final.transpose_and_contiguous_tensor"):
            model_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocessing_total = (time.perf_counter() - started) * 1000.0

        inference_timings: dict[str, float] = {}
        inference_started = time.perf_counter()
        with Timer(inference_timings, "waviness_inference.v18_regressor_onnxruntime"):
            output = server.REGRESSOR_SESSION.run(["scores"], {"images": model_tensor})[0]
        with Timer(inference_timings, "waviness_inference.score_extraction"):
            score = float(output.reshape(-1)[0])
        inference_total = (time.perf_counter() - inference_started) * 1000.0
        timings.update(inference_timings)

        return {
            "prediction": score,
            "preprocessing_total_ms": preprocessing_total,
            "waviness_inference_total_ms": inference_total,
            "total_raw_to_score_ms": (time.perf_counter() - started) * 1000.0,
            "timings_ms": timings,
            "image": {"width": width, "height": height, "format": photo.suffix.lower()},
            "intermediate": {
                "step1_water_bbox_xywh": [left, top, box_width, box_height],
                "regressor_tensor_shape": list(model_tensor.shape),
                "regressor_tensor_dtype": str(model_tensor.dtype),
            },
        }
    finally:
        if isinstance(image, server.NativeRGBFrame):
            image.close()
