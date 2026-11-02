#include <libheif/heif.h>
#include <libheif/heif_aux_images.h>
#include <libheif/heif_tiling.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void set_error(char* out, size_t size, const char* message) {
  if (out && size) std::snprintf(out, size, "%s", message);
}

extern "C" int wave_heif_decode_thumbnail_rgb(
    const char* path, unsigned char** output, int* out_w, int* out_h,
    int* source_w, int* source_h, char* message, size_t message_size) {
  if (!path || !output || !out_w || !out_h || !source_w || !source_h) return 1;
  *output = nullptr;
  heif_context* context = heif_context_alloc();
  if (!context) return 2;
  auto error = heif_context_read_from_file(context, path, nullptr);
  heif_image_handle* primary = nullptr;
  if (!error.code) error = heif_context_get_primary_image_handle(context, &primary);
  if (error.code || !primary) {
    set_error(message, message_size, error.message ? error.message : "primary image unavailable");
    if (primary) heif_image_handle_release(primary);
    heif_context_free(context);
    return 3;
  }
  *source_w = heif_image_handle_get_width(primary);
  *source_h = heif_image_handle_get_height(primary);
  const int count = heif_image_handle_get_number_of_thumbnails(primary);
  if (count < 1) {
    set_error(message, message_size, "no embedded thumbnail");
    heif_image_handle_release(primary);
    heif_context_free(context);
    return 4;
  }
  heif_item_id id = 0;
  heif_image_handle_get_list_of_thumbnail_IDs(primary, &id, 1);
  heif_image_handle* thumbnail = nullptr;
  error = heif_image_handle_get_thumbnail(primary, id, &thumbnail);
  heif_image* decoded = nullptr;
  if (!error.code) error = heif_decode_image(thumbnail, &decoded, heif_colorspace_RGB,
      heif_chroma_interleaved_RGB, nullptr);
  if (error.code || !decoded) {
    set_error(message, message_size, error.message ? error.message : "thumbnail decode failed");
    if (decoded) heif_image_release(decoded);
    if (thumbnail) heif_image_handle_release(thumbnail);
    heif_image_handle_release(primary);
    heif_context_free(context);
    return 5;
  }
  *out_w = heif_image_get_width(decoded, heif_channel_interleaved);
  *out_h = heif_image_get_height(decoded, heif_channel_interleaved);
  int stride = 0;
  const auto* source = heif_image_get_plane_readonly(decoded, heif_channel_interleaved, &stride);
  const size_t row_bytes = static_cast<size_t>(*out_w) * 3;
  *output = static_cast<unsigned char*>(std::malloc(row_bytes * *out_h));
  if (!*output) {
    heif_image_release(decoded);
    heif_image_handle_release(thumbnail);
    heif_image_handle_release(primary);
    heif_context_free(context);
    return 6;
  }
  for (int y = 0; y < *out_h; ++y)
    std::memcpy(*output + static_cast<size_t>(y) * row_bytes,
                source + static_cast<size_t>(y) * stride, row_bytes);
  heif_image_release(decoded);
  heif_image_handle_release(thumbnail);
  heif_image_handle_release(primary);
  heif_context_free(context);
  return 0;
}

extern "C" int wave_heif_decode_region_rgb(
    const char* path, int x0, int y0, int x1, int y1, int workers,
    unsigned char** output, int* out_w, int* out_h, char* message, size_t message_size) {
  if (!path || !output || !out_w || !out_h || workers < 1 || x1 <= x0 || y1 <= y0) return 1;
  *output = nullptr;
  heif_context* context = heif_context_alloc();
  if (!context) return 2;
  auto error = heif_context_read_from_file(context, path, nullptr);
  heif_image_handle* handle = nullptr;
  if (!error.code) error = heif_context_get_primary_image_handle(context, &handle);
  if (error.code || !handle) {
    set_error(message, message_size, error.message ? error.message : "primary image unavailable");
    if (handle) heif_image_handle_release(handle);
    heif_context_free(context);
    return 3;
  }
  const int image_w = heif_image_handle_get_width(handle);
  const int image_h = heif_image_handle_get_height(handle);
  x0 = std::clamp(x0, 0, image_w); x1 = std::clamp(x1, 0, image_w);
  y0 = std::clamp(y0, 0, image_h); y1 = std::clamp(y1, 0, image_h);
  if (x1 <= x0 || y1 <= y0) {
    set_error(message, message_size, "requested region is outside the primary image");
    heif_image_handle_release(handle); heif_context_free(context); return 4;
  }
  heif_image_tiling tiling{}; tiling.version = 1;
  error = heif_image_handle_get_image_tiling(handle, 1, &tiling);
  if (error.code || !tiling.tile_width || !tiling.tile_height) {
    set_error(message, message_size, "primary image is not tiled");
    heif_image_handle_release(handle); heif_context_free(context); return 5;
  }
  const int sx = tiling.left_offset ? static_cast<int>(tiling.left_offset) - static_cast<int>(tiling.tile_width) : 0;
  const int sy = tiling.top_offset ? static_cast<int>(tiling.top_offset) - static_cast<int>(tiling.tile_height) : 0;
  *out_w = x1 - x0; *out_h = y1 - y0;
  const size_t row_bytes = static_cast<size_t>(*out_w) * 3;
  *output = static_cast<unsigned char*>(std::calloc(static_cast<size_t>(*out_h), row_bytes));
  if (!*output) { heif_image_handle_release(handle); heif_context_free(context); return 6; }
  const int first_tx = std::max(0, (x0 - sx) / static_cast<int>(tiling.tile_width));
  const int last_tx = std::min(static_cast<int>(tiling.num_columns) - 1,
      std::max(0, (x1 - 1 - sx) / static_cast<int>(tiling.tile_width)));
  const int first_ty = std::max(0, (y0 - sy) / static_cast<int>(tiling.tile_height));
  const int last_ty = std::min(static_cast<int>(tiling.num_rows) - 1,
      std::max(0, (y1 - 1 - sy) / static_cast<int>(tiling.tile_height)));
  struct Task { int x; int y; };
  std::vector<Task> tasks;
  for (int ty = first_ty; ty <= last_ty; ++ty)
    for (int tx = first_tx; tx <= last_tx; ++tx) tasks.push_back({tx, ty});
  heif_image_handle_release(handle);
  heif_context_free(context);
  const std::string filename(path);
  std::atomic<size_t> next{0};
  std::atomic<int> failed{0};
  std::vector<std::thread> threads;
  const int thread_count = std::min(workers, static_cast<int>(tasks.size()));
  for (int worker = 0; worker < thread_count; ++worker) {
    threads.emplace_back([&] {
      heif_context* worker_context = heif_context_alloc();
      if (!worker_context) { failed.store(8); return; }
      auto read_error = heif_context_read_from_file(worker_context, filename.c_str(), nullptr);
      heif_image_handle* worker_handle = nullptr;
      if (read_error.code || heif_context_get_primary_image_handle(worker_context, &worker_handle).code) {
        failed.store(9); heif_context_free(worker_context); return;
      }
      heif_decoding_options* options = heif_decoding_options_alloc();
      options->version = 8; options->num_codec_threads = 1;
      for (;;) {
        const size_t index = next.fetch_add(1);
        if (index >= tasks.size() || failed.load()) break;
        const int tx = tasks[index].x, ty = tasks[index].y;
        heif_image* tile = nullptr;
        auto tile_error = heif_image_handle_decode_image_tile(worker_handle, &tile,
            heif_colorspace_RGB, heif_chroma_interleaved_RGB, options,
            static_cast<unsigned int>(tx), static_cast<unsigned int>(ty));
        if (tile_error.code || !tile) {
          failed.store(tile_error.code ? tile_error.code : 7);
          if (tile) heif_image_release(tile);
          break;
        }
        const int tile_x = tx * static_cast<int>(tiling.tile_width) + sx;
        const int tile_y = ty * static_cast<int>(tiling.tile_height) + sy;
        const int tile_w = heif_image_get_width(tile, heif_channel_interleaved);
        const int tile_h = heif_image_get_height(tile, heif_channel_interleaved);
        const int left = std::max(x0, tile_x), top = std::max(y0, tile_y);
        const int right = std::min(x1, tile_x + tile_w), bottom = std::min(y1, tile_y + tile_h);
        int stride = 0;
        const auto* pixels = heif_image_get_plane_readonly(tile, heif_channel_interleaved, &stride);
        for (int y = top; y < bottom; ++y) {
          const auto* src = pixels + static_cast<size_t>(y - tile_y) * stride + static_cast<size_t>(left - tile_x) * 3;
          auto* dst = *output + static_cast<size_t>(y - y0) * row_bytes + static_cast<size_t>(left - x0) * 3;
          std::memcpy(dst, src, static_cast<size_t>(right - left) * 3);
        }
        heif_image_release(tile);
      }
      heif_decoding_options_free(options);
      heif_image_handle_release(worker_handle);
      heif_context_free(worker_context);
    });
  }
  for (auto& thread : threads) thread.join();
  if (failed.load()) {
    set_error(message, message_size, "parallel tile decode failed");
    std::free(*output); *output = nullptr; return 7;
  }
  return 0;
}

extern "C" void wave_heif_free_region(void* buffer) { std::free(buffer); }
