from pathlib import Path

from storm_zero_llm.config import StormZeroConfig


def test_config_loads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_HOST = 0.0.0.0",
                "LLM_PORT = 9000",
                "PM2_APP = sz-llm",
                "CUSTOM_MODELS_PATH = custom_models",
                "DEDICATED_MODELS_PATH = dedicated_models",
                "BASE_MODEL = dedicated_models/text.gguf",
                "DETECTION_MODEL = dedicated_models/detection.gguf",
                "POWER_BASE_MODEL = dedicated_models/power-base.gguf",
                "POWER_DETECTION_MODEL = dedicated_models/power-detection.gguf",
                "TTS_PATH = tts_assets",
                "GENERATE_IMAGE_MODEL = black-forest-labs/FLUX.1-dev",
                "GENERATE_IMAGE_LORA_MODEL = lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
                "IMAGE_DEVICE = mps",
                "IMAGE_CPU_OFFLOAD = true",
                "IMAGE_SEQUENTIAL_CPU_OFFLOAD = false",
                "IMAGE_COREML_PATH = models/coreml-sd",
                "IMAGE_COMPUTE_UNIT = CPU_AND_NE",
                "IMAGE_NUM_INFERENCE_STEPS = 25",
                "IMAGE_GUIDANCE_SCALE = 8.5",
                "HG_TOKEN = hf_test_token_123",
                "MUSIC_MODEL = dedicated_models/music.gguf",
                "VIDEO_MODEL = dedicated_models/video.gguf",
                "LLM_N_BATCH = 1024",
                "LLM_FLASH_ATTN = false",
            ]
        ),
        encoding="utf-8",
    )

    config = StormZeroConfig.load(project_root=tmp_path)

    assert config.llm_host == "0.0.0.0"
    assert config.llm_port == 9000
    assert config.pm2_app == "sz-llm"
    assert config.base_model == (tmp_path / "dedicated_models" / "text.gguf").resolve()
    assert config.detection_model == (tmp_path / "dedicated_models" / "detection.gguf").resolve()
    assert config.power_base_model == (tmp_path / "dedicated_models" / "power-base.gguf").resolve()
    assert config.power_detection_model == (tmp_path / "dedicated_models" / "power-detection.gguf").resolve()
    assert config.tts_path == (tmp_path / "tts_assets").resolve()
    assert config.generate_image_model == "black-forest-labs/FLUX.1-dev"
    assert config.generate_image_lora_model == "lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1"
    assert config.image_device == "mps"
    assert config.image_cpu_offload is True
    assert config.image_sequential_cpu_offload is False
    assert config.image_coreml_path == (tmp_path / "models" / "coreml-sd").resolve()
    assert config.image_compute_unit == "CPU_AND_NE"
    assert config.image_num_inference_steps == 25
    assert config.image_guidance_scale == 8.5
    assert config.huggingface_token == "hf_test_token_123"
    assert config.music_model == (tmp_path / "dedicated_models" / "music.gguf").resolve()
    assert config.video_model == (tmp_path / "dedicated_models" / "video.gguf").resolve()
    assert config.llm_runtime.n_ctx == 8192
    assert config.llm_runtime.n_batch == 1024
    assert config.llm_runtime.flash_attn is False
