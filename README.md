# ComfyUI Quick Watermark

A small ComfyUI custom node that applies one PNG watermark image across every frame in an `IMAGE` batch. Video workflows usually pass frames as a batched `IMAGE`, so this works for still images and videos.

## Node

`Quick Watermark Video` is available under `image/watermark`.

Inputs:

- `image`: the source image or video frame batch.
- `watermark_preset`: built-in watermark to use when no `watermark` input is connected. Options are `speaker-white` and `speaker-black`.
- `watermark` optional: an image from `Load Image`, usually a PNG logo or watermark. When connected, this overrides `watermark_preset`.
- `alignment`: watermark placement. Defaults to `bottom-right`.
- `offset_percentage`: margin from the selected edge, based on the source frame size.
- `resize_percentage`: watermark width as a percentage of the source frame width.
- `opacity`: global watermark opacity from `0` to `100`. Defaults to `60`.
- `watermark_mask` optional: connect the `mask` output from ComfyUI `Load Image` to preserve PNG alpha transparency.

The node returns the watermarked `IMAGE` batch.

## PNG Alpha

ComfyUI's `Load Image` node separates PNG alpha into its `mask` output. To preserve transparent watermarks, connect both:

- `Load Image.image` to `watermark`
- `Load Image.mask` to `watermark_mask`

The built-in presets use their PNG alpha automatically. If a custom `watermark` image is connected with no mask, the watermark image is treated as fully opaque before the `opacity` setting is applied.
