"""Configuration loading for Storm Zero LLM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _resolve_path(root: Path, value: str | None, default: Path) -> Path:
    raw = Path(value).expanduser() if value else default
    if not raw.is_absolute():
        raw = (root / raw).resolve()
    return raw


def _resolve_model_ref(root: Path, value: str | None, default: str) -> str:
    raw = (value or default).strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        rooted = (root / candidate).resolve()
        if rooted.exists():
            return str(rooted)
    if candidate.exists():
        return str(candidate.resolve())
    return raw


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(frozen=True)
class LLMRuntimeConfig:
    n_gpu_layers: int
    n_threads: int
    n_ctx: int
    n_batch: int = 512
    flash_attn: bool = True


@dataclass(frozen=True)
class StormZeroConfig:
    project_root: Path
    llm_host: str
    llm_port: int
    pm2_app: str
    custom_models_path: Path
    dedicated_models_path: Path
    base_model: Path
    detection_model: Path
    power_base_model: Path
    power_detection_model: Path
    generate_image_model: str
    generate_image_lora_model: str | None
    image_coreml_path: Path | None
    image_compute_unit: str
    image_num_inference_steps: int
    image_guidance_scale: float
    image_device: str | None
    image_cpu_offload: bool
    image_sequential_cpu_offload: bool
    huggingface_token: str | None
    music_model: Path
    video_model: Path
    readme_path: Path
    memory_root: Path
    tts_path: Path
    database: DatabaseConfig | None
    llm_runtime: LLMRuntimeConfig

    @classmethod
    def load(cls, project_root: str | Path | None = None) -> "StormZeroConfig":
        root = Path(project_root or Path.cwd()).resolve()
        env_values = read_env_file(root / ".env")
        merged = {**env_values, **os.environ}

        llm_host = merged.get("LLM_HOST", "127.0.0.1")
        llm_port = _parse_int(merged.get("LLM_PORT") or merged.get("SERVER_PORT") or merged.get("PORT"), 8765)
        pm2_app = merged.get("PM2_APP", "sz-llm")

        custom_models_path = _resolve_path(root, merged.get("CUSTOM_MODELS_PATH"), root / "custom_models")
        dedicated_models_path = _resolve_path(root, merged.get("DEDICATED_MODELS_PATH"), root / "dedicated_models")

        base_model_raw = merged.get("BASE_MODEL") or merged.get("BASE_MODEL_PATH")
        if base_model_raw:
            base_model = _resolve_path(root, base_model_raw, dedicated_models_path / "base.gguf")
        else:
            base_model = dedicated_models_path / "base.gguf"

        detection_model = _resolve_path(
            root,
            merged.get("DETECTION_MODEL"),
            dedicated_models_path / "detection.gguf",
        )

        power_base_model_raw = merged.get("POWER_BASE_MODEL") or merged.get("POWER_BASE_MODEL_PATH")
        if power_base_model_raw:
            power_base_model = _resolve_path(root, power_base_model_raw, dedicated_models_path / "power-base.gguf")
        else:
            power_base_model = dedicated_models_path / "power-base.gguf"

        power_detection_model = _resolve_path(
            root,
            merged.get("POWER_DETECTION_MODEL"),
            dedicated_models_path / "power-detection.gguf",
        )
        generate_image_model = _resolve_model_ref(
            root,
            merged.get("GENERATE_IMAGE_MODEL"),
            "black-forest-labs/FLUX.1-dev",
        )
        generate_image_lora_raw = merged.get("GENERATE_IMAGE_LORA_MODEL")
        generate_image_lora_model = (
            None
            if generate_image_lora_raw is not None and not generate_image_lora_raw.strip()
            else _resolve_model_ref(
                root,
                generate_image_lora_raw,
                "lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
            )
        )
        image_coreml_raw = merged.get("IMAGE_COREML_PATH")
        image_coreml_path = (
            _resolve_path(root, image_coreml_raw, root / "models" / "coreml-stable-diffusion")
            if image_coreml_raw
            else None
        )
        image_compute_unit = (merged.get("IMAGE_COMPUTE_UNIT") or "CPU_AND_GPU").strip()
        image_num_inference_steps = _parse_int(merged.get("IMAGE_NUM_INFERENCE_STEPS"), 20)
        image_guidance_scale = _parse_float(merged.get("IMAGE_GUIDANCE_SCALE"), 7.5)
        image_device = merged.get("IMAGE_DEVICE") or None
        image_cpu_offload = _parse_bool(merged.get("IMAGE_CPU_OFFLOAD"), default=True)
        image_sequential_cpu_offload = _parse_bool(
            merged.get("IMAGE_SEQUENTIAL_CPU_OFFLOAD"),
            default=False,
        )
        huggingface_token = (merged.get("HG_TOKEN") or merged.get("HF_TOKEN") or "").strip() or None
        if huggingface_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = huggingface_token
        music_model = _resolve_path(root, merged.get("MUSIC_MODEL"), dedicated_models_path / "generate-music.gguf")
        video_model = _resolve_path(root, merged.get("VIDEO_MODEL"), dedicated_models_path / "generate-video.gguf")

        memory_root = _resolve_path(root, merged.get("MEMORY_ROOT"), root / "data" / "memory")
        memory_root.mkdir(parents=True, exist_ok=True)
        tts_path = _resolve_path(root, merged.get("TTS_PATH"), root / "data" / "tts")

        db_host = merged.get("DB_HOST")
        db_name = merged.get("DB_NAME")
        db_user = merged.get("DB_USER")
        db_pass = merged.get("DB_PASS")
        db_port = _parse_int(merged.get("DB_PORT"), 3306)
        database = None
        if db_host and db_name and db_user and db_pass:
            database = DatabaseConfig(
                host=db_host,
                port=db_port,
                name=db_name,
                user=db_user,
                password=db_pass,
            )

        llm_runtime = LLMRuntimeConfig(
            n_gpu_layers=_parse_int(merged.get("LLM_N_GPU_LAYERS"), -1),
            n_threads=_parse_int(merged.get("LLM_N_THREADS"), 8),
            n_ctx=_parse_int(merged.get("LLM_N_CTX"), 8192),
            n_batch=_parse_int(merged.get("LLM_N_BATCH"), 512),
            flash_attn=_parse_bool(merged.get("LLM_FLASH_ATTN"), True),
        )

        return cls(
            project_root=root,
            llm_host=llm_host,
            llm_port=llm_port,
            pm2_app=pm2_app,
            custom_models_path=custom_models_path,
            dedicated_models_path=dedicated_models_path,
            base_model=base_model,
            detection_model=detection_model,
            power_base_model=power_base_model,
            power_detection_model=power_detection_model,
            generate_image_model=generate_image_model,
            generate_image_lora_model=generate_image_lora_model,
            image_coreml_path=image_coreml_path,
            image_compute_unit=image_compute_unit,
            image_num_inference_steps=image_num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            image_device=image_device,
            image_cpu_offload=image_cpu_offload,
            image_sequential_cpu_offload=image_sequential_cpu_offload,
            huggingface_token=huggingface_token,
            music_model=music_model,
            video_model=video_model,
            readme_path=root / "README.md",
            memory_root=memory_root,
            tts_path=tts_path,
            database=database,
            llm_runtime=llm_runtime,
        )
