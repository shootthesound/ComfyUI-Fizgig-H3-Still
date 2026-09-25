"""Fizgig H3 Still: a true single-frame latent for MiniMax H3, the way Fizgig's previews and still training render.

ComfyUI's own H3 latents start at 5 frames (2 latent frames) and snap to the 17k+5 grid. H3's native image
convention is ONE latent frame (the reference VAE and model special-case it), which is what Fizgig's previews
render. Plug this in place of the latent output of "MiniMax H3 Image to Video"
(keep its conditioning); decode with Fizgig H3 Still Decode (the stock VAE Decode bands a lone frame).
"""
import torch
import comfy.model_management
import comfy.nested_tensor

FPS, AUDIO_LATENT_FPS = 24, 40


class FizgigH3StillLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32,
                              "tooltip": "Image width in pixels (multiple of 32)."}),
            "height": ("INT", {"default": 1344, "min": 64, "max": 4096, "step": 32,
                               "tooltip": "Image height in pixels (multiple of 32)."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make"
    CATEGORY = "Fizgig"
    DESCRIPTION = ("One-frame MiniMax H3 latent (a still), as Fizgig renders its previews. "
                   "Use its output instead of the latent from MiniMax H3 Image to Video.")

    def make(self, width, height, batch_size):
        dev = comfy.model_management.intermediate_device()
        # the DiT patchifies 2x2 on a 16x latent grid, so the latent size must be even
        lh, lw = (height // 16) // 2 * 2, (width // 16) // 2 * 2
        video = torch.zeros([batch_size, 24, 1, lh, lw], device=dev)
        audio_t = max(1, round(1 / FPS * AUDIO_LATENT_FPS))       # 2, as Fizgig's sampler uses for one frame
        audio = torch.zeros([batch_size, 32, 2, audio_t], device=dev)
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


class FizgigH3StillDecode:
    """VAE Decode for H3 stills, the way Fizgig's previews decode them.

    H3's ViT decoder was trained on 5-latent temporal groups, and its token coordinates are normalised over the
    latent's T. A lone latent token (what a still is, and what the stock VAE Decode feeds it) sits outside that
    regime and comes back banded and dark. Fizgig replicates the frame into a full 5-latent group, decodes that
    (spatially tiled as usual — tiling is not optional above 256 px) and keeps pixel frame 3, just past the
    decoder's causal lead-in: 30 dB round-trip vs 17 dB for the lone token (Fizgig vae.py, single_frame_mode
    "group"). This node does exactly that. Clips (more than one latent frame) go through the stock decode."""

    GROUP, KEEP = 5, 3

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Fizgig"
    DESCRIPTION = ("Decodes an H3 still the way Fizgig's previews do (the frame replicated into a full 5-frame group), "
                   "without the banding of the stock VAE Decode on a single frame. Use it in place of VAE Decode.")

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        fsm = vae.first_stage_model
        if latent.ndim != 5 or latent.shape[2] != 1 or not hasattr(fsm, "_adaptive_decode"):
            images = vae.decode(latent)          # clips / other VAEs: the stock path
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)
        group_shape = (1, latent.shape[1], self.GROUP, latent.shape[3], latent.shape[4])
        mem = vae.memory_used_decode(group_shape, vae.vae_dtype)
        comfy.model_management.load_models_gpu([vae.patcher], memory_required=mem,
                                               force_full_load=getattr(vae, "disable_offload", False))
        out = []
        with torch.no_grad():
            for b in range(latent.shape[0]):
                z = latent[b:b + 1].to(device=vae.device, dtype=vae.vae_dtype)
                lm = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
                ls = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
                zz = (z * ls + lm).repeat(1, 1, self.GROUP, 1, 1)
                raw = fsm._adaptive_decode(zz)                      # [1, 3, 20, H, W], tiled as usual
                px = fsm._finalize_pixels(raw[:, :, self.KEEP:self.KEEP + 1])   # [1, 3, 1, H, W] in [0, 1]
                out.append(px[:, :, 0].movedim(1, -1).to(comfy.model_management.intermediate_device()))
                del raw, zz
        return (torch.cat(out),)


NODE_CLASS_MAPPINGS = {"FizgigH3StillLatent": FizgigH3StillLatent, "FizgigH3StillDecode": FizgigH3StillDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"FizgigH3StillLatent": "Fizgig H3 Still Latent (single frame)",
                              "FizgigH3StillDecode": "Fizgig H3 Still Decode"}
