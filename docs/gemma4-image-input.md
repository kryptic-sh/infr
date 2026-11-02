# Gemma 4 image input (PR #20 integration)

Gemma 4 image input is available through `infr serve`'s OpenAI-compatible
`messages[].content` image parts on the ordinary Vulkan backend. The vision tower
runs on the CPU; the language model uses Vulkan. Configure the matching Gemma 4
`gemma4v` projector with `serve.mmproj` / `INFR_MMPROJ` (or place the automatically
recognized `mmproj*.gguf` beside the model). CPU, Metal, and MTP generator paths
currently reject image requests before committing response headers.

Supported inputs are base64 `data:` image URIs. Remote image URLs are not fetched.
`serve.image_max_tokens` / `INFR_IMAGE_MAX_TOKENS` defaults to 280 per image.

## Resource and correctness contracts

- At most **4 images** per request; at most **16 MiB** of decoded file bytes per
  image. Base64 length is checked before allocating the decoded file.
- Image dimensions are checked before decoding: at most 16,384 pixels per side,
  16 Mi pixels per image, and 32 Mi pixels across the request. The decoder has an
  explicit 128 MiB allocation limit.
- A separate process-wide admission gate permits **one image preflight decode at
  a time**, independently of generation/GPU slots. Overload returns HTTP 503
  without queuing decoder jobs. The blocking closure owns the permit, including
  when the client disconnects.
- The aligned resize is strictly bounded by the configured soft-token ceiling
  and the projector's position table, including tiny images and extreme aspect
  ratios. The complete request is capped at 1,120 image soft tokens; its text,
  markers, and image tokens must fit the session context before tower execution.
- Projector metadata and tensor geometry are validated at open, including
  Q/K/V widths, normalization vectors, patch/position tensors, projection shapes,
  and supported float tensor byte counts. Runtime bounds are checked before unsafe
  GEMM calls. Cancellation is checked between images and tower layers.
- Encoder-cache hits require identical complete image bytes and the same token
  limit. The compact token tag is **not** an image identity: KV-prefix reuse stops
  before media tokens, even when their tags compare equal.
- Invalid payloads and unsupported models/backends return HTTP 400 JSON before
  SSE headers. Backend validation does not run the vision transformer or acquire
  a generation slot.

## Verification

The integration adds CPU regressions for malformed geometry/attention inputs,
tiny/extreme-aspect resizing, payload validation, media-tag prefix collisions,
request count/byte limits, and HTTP rejection before streaming headers.

A headless end-to-end smoke test used Gemma 4 E4B Q4_K_M and its F16 projector on
an AMD Ryzen 7 7800X3D integrated GPU, under `VK_LAYER_KHRONOS_validation`:

- One slot, context 256, ubatch 16, greedy sampling, thinking disabled, image-token
  ceiling 70.
- Two consecutive requests with identical text asking the dominant color and
  different solid 64×64 PNGs returned **Red**, then **Blue**. Both images were
  resized to 384×384 (64 soft tokens), exercising changed-image prefix reuse.
- Invalid image bytes returned HTTP 400 JSON with both `stream=false` and
  `stream=true`.
- No Vulkan validation errors or warnings were emitted.

This is a semantic/integration smoke test, not a numerical encoder golden against
Hugging Face or llama.cpp. The MMQ/decode benchmarks and their hardware limitations
are documented separately in `perf/vulkan-q4k-packed-meta.md`.
