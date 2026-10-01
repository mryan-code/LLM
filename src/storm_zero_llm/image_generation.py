"""Image generation via Core ML Stable Diffusion or Flux (diffusers)."""

from __future__ import annotations

import base64
import io
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ImageGenerator(Protocol):
    def generate(self, prompt: str) -> dict[str, str]: ...


@dataclass(frozen=True)
class ImageGenerationConfig:
    flux_model: str
    coreml_path: Path | None = None
    lora_model: str | None = None
    device: str | None = None
    token: str | None = None
    cpu_offload: bool = True
    sequential_cpu_offload: bool = False
    compute_unit: str = "CPU_AND_GPU"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5


def create_image_generator(config: ImageGenerationConfig) -> ImageGenerator:
    if config.coreml_path is not None:
        return CoreMLImageGenerator(config)
    return FluxImageGenerator(config)


def _encode_pil_image(image) -> dict[str, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "mime_type": "image/png",
        "base64": encoded,
    }


_CLIP_MAX_LENGTH = 77


def _clip_token_ids(tokenizer: object, text: str, *, truncation: bool, max_length: int | None = None) -> list[int]:
    encode = getattr(tokenizer, "__call__", None)
    if encode is None:
        return []
    kwargs: dict[str, object] = {
        "truncation": truncation,
        "add_special_tokens": True,
        "return_tensors": None,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = encode(text, **kwargs)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, (tuple, list)) and input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = list(input_ids[0])
    elif isinstance(input_ids, (tuple, list)):
        input_ids = list(input_ids)
    else:
        return []
    return [int(token_id) for token_id in input_ids]


def truncate_prompt_for_clip(prompt: str, tokenizer: object, max_length: int = _CLIP_MAX_LENGTH) -> str:
    """Keep prompt within CLIP's fixed context so Core ML text encoder shapes match.

    Trims the original string from the end (keeps the user request at the start)
    instead of decoding token ids back to text, which can collapse to empty.
    """
    text = prompt.strip()
    if not text:
        return text
    try:
        if len(_clip_token_ids(tokenizer, text, truncation=False)) <= max_length:
            return text
    except Exception:
        return text[:240]

    low = 0
    high = len(text)
    best = text[: min(len(text), 240)]
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip()
        if not candidate:
            low = mid + 1
            continue
        try:
            length = len(_clip_token_ids(tokenizer, candidate, truncation=False))
        except Exception:
            return best
        if length <= max_length:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or text[:240]


def _patch_coreml_pipeline_imports() -> None:
    """Patch ml-stable-diffusion for transformers v5 and a numpy truth-value bug."""
    try:
        import python_coreml_stable_diffusion
    except ImportError:
        return

    pipeline = Path(python_coreml_stable_diffusion.__file__).parent / "pipeline.py"
    if not pipeline.is_file():
        return

    text = pipeline.read_text(encoding="utf-8")
    original = text

    old_import = "from transformers import CLIPFeatureExtractor, CLIPTokenizer"
    new_import = """from transformers import CLIPTokenizer
try:
    from transformers import CLIPFeatureExtractor
except ImportError:
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor as CLIPFeatureExtractor"""
    if old_import in text:
        text = text.replace(old_import, new_import)

    # `not np.equal(a, b)` raises on multi-element arrays; compare with array_equal.
    old_truncation_check = """            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.equal(
                    text_input_ids, untruncated_ids
            ):"""
    new_truncation_check = """            if untruncated_ids.shape[-1] > text_input_ids.shape[-1] or (
                    untruncated_ids.shape == text_input_ids.shape
                    and not np.array_equal(text_input_ids, untruncated_ids)
            ):"""
    if old_truncation_check in text:
        text = text.replace(old_truncation_check, new_truncation_check)

    if text != original:
        pipeline.write_text(text, encoding="utf-8")


class CoreMLImageGenerator:
    _lock = threading.Lock()

    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self._pipe = None

    def generate(self, prompt: str) -> dict[str, str]:
        if sys.platform != "darwin":
            raise RuntimeError(
                "Core ML image generation requires macOS. "
                "Unset IMAGE_COREML_PATH to use the Flux backend instead."
            )

        with self._lock:
            if self._pipe is None:
                self._pipe = self._load_pipe()
            tokenizer = getattr(self._pipe, "tokenizer", None)
            prompt_to_use = prompt
            if tokenizer is not None:
                prompt_to_use = truncate_prompt_for_clip(prompt, tokenizer)
            try:
                result = self._pipe(
                    prompt=prompt_to_use,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                )
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                if "broadcast together with shapes" in message or "ambiguous" in message.lower():
                    raise RuntimeError(
                        "Core ML image generation failed because the prompt is incompatible "
                        f"with the CLIP {_CLIP_MAX_LENGTH}-token text encoder. "
                        "Shorten the image request or reduce appended user/conversation context."
                    ) from exc
                if "Error computing NN outputs" in message or "out of memory" in message.lower():
                    raise RuntimeError(
                        "Core ML image generation failed due to memory pressure. "
                        "Try IMAGE_COMPUTE_UNIT=CPU_AND_NE, reduce LLM_N_GPU_LAYERS, "
                        "or use a smaller Core ML model variant."
                    ) from exc
                raise

        image = result.images[0]
        return _encode_pil_image(image)

    def _load_pipe(self):
        _patch_coreml_pipeline_imports()

        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        from python_coreml_stable_diffusion.coreml_model import get_resource_type
        from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

        model_version = self.config.flux_model
        coreml_path = self.config.coreml_path
        if coreml_path is None:
            raise RuntimeError("IMAGE_COREML_PATH is required for Core ML image generation.")

        if not coreml_path.is_dir():
            raise RuntimeError(f"Core ML model directory does not exist: {coreml_path}")

        sources = get_resource_type(str(coreml_path))
        is_xl = "xl" in model_version.lower()
        pipe_class = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

        load_kwargs: dict[str, object] = {
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if self.config.token:
            load_kwargs["token"] = self.config.token

        pytorch_pipe = pipe_class.from_pretrained(model_version, **load_kwargs)
        force_zeros = bool(getattr(pytorch_pipe, "config", {}).get("force_zeros_for_empty_prompt", True))

        return get_coreml_pipe(
            pytorch_pipe=pytorch_pipe,
            mlpackages_dir=str(coreml_path),
            model_version=model_version,
            compute_unit=self.config.compute_unit,
            delete_original_pipe=True,
            force_zeros_for_empty_prompt=force_zeros,
            sources=sources,
        )


class FluxImageGenerator:
    _lock = threading.Lock()

    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self._pipe = None
        self._device_in_use: str | None = None

    def generate(self, prompt: str) -> dict[str, str]:
        with self._lock:
            if self._pipe is None:
                self._pipe = self._load_pipe()
            self._release_transient_memory()
            try:
                image = self._pipe(prompt).images[0]
            except RuntimeError as exc:
                if "MPS backend out of memory" in str(exc):
                    raise RuntimeError(
                        "Image generation exceeded available Apple GPU memory. "
                        "Keep IMAGE_CPU_OFFLOAD=true (default), set IMAGE_SEQUENTIAL_CPU_OFFLOAD=true "
                        "for tighter limits, reduce LLM_N_GPU_LAYERS so chat models leave Metal headroom, "
                        "or set IMAGE_DEVICE=cpu."
                    ) from exc
                raise
            finally:
                self._release_transient_memory()

        return _encode_pil_image(image)

    def _resolve_device(self) -> str:
        if self.config.device:
            return self.config.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_pipe(self):
        import torch
        from diffusers import DiffusionPipeline

        device = self._resolve_device()
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

        load_kwargs: dict[str, object] = {"torch_dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "cuda"
        if self.config.token:
            load_kwargs["token"] = self.config.token

        pipe = DiffusionPipeline.from_pretrained(self.config.flux_model, **load_kwargs)
        if self.config.lora_model:
            try:
                import peft  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "PEFT is required for LoRA image generation. "
                    "Install image dependencies with: scripts/install.sh runtime"
                ) from exc

            lora_kwargs: dict[str, object] = {}
            if self.config.token:
                lora_kwargs["token"] = self.config.token
            try:
                pipe.load_lora_weights(self.config.lora_model, **lora_kwargs)
            except ValueError as exc:
                if "PEFT backend is required" in str(exc):
                    raise RuntimeError(
                        "PEFT is required for LoRA image generation. "
                        "Install image dependencies with: scripts/install.sh runtime"
                    ) from exc
                raise

        if device == "cuda":
            pass
        elif device == "mps":
            if self.config.sequential_cpu_offload:
                pipe.enable_sequential_cpu_offload()
            elif self.config.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)

        vae = getattr(pipe, "vae", None)
        if vae is not None:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

        self._device_in_use = device
        return pipe

    def _release_transient_memory(self) -> None:
        if self._device_in_use != "mps":
            return
        try:
            import torch
        except ImportError:
            return
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
