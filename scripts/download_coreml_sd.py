#!/usr/bin/env python3
"""Download Apple Core ML Stable Diffusion weights for Storm Zero LLM."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

VARIANTS = {
    "urpm-v13-split-einsum": {
        "repo_id": "coreml-community/coreml-URPM-v13",
        "zip_name": "split_einsum/URPM-v13_split-einsum.zip",
        "compute_unit": "CPU_AND_NE",
        "model_version": "runwayml/stable-diffusion-v1-5",
        "output_dir_name": "coreml-urpm-v13",
        "description": "URPM v13 uncensored SD1.5, split_einsum (Neural Engine)",
    },
    "urpm-v13-original": {
        "repo_id": "coreml-community/coreml-URPM-v13",
        "zip_name": "original/URPM-v13_original.zip",
        "compute_unit": "CPU_AND_GPU",
        "model_version": "runwayml/stable-diffusion-v1-5",
        "output_dir_name": "coreml-urpm-v13",
        "description": "URPM v13 uncensored SD1.5, original attention (CPU/GPU)",
    },
    "palettized-original": {
        "repo_id": "apple/coreml-stable-diffusion-2-1-base-palettized",
        "zip_name": "coreml-stable-diffusion-2-1-base-palettized_original_compiled.zip",
        "compute_unit": "CPU_AND_GPU",
        "model_version": "stabilityai/stable-diffusion-2-1-base",
        "output_dir_name": "coreml-stable-diffusion",
        "description": "6-bit palettized, original attention (recommended for Mac mini)",
    },
    "palettized-split-einsum-v2": {
        "repo_id": "apple/coreml-stable-diffusion-2-1-base-palettized",
        "zip_name": "coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip",
        "compute_unit": "CPU_AND_NE",
        "model_version": "stabilityai/stable-diffusion-2-1-base",
        "output_dir_name": "coreml-stable-diffusion",
        "description": "6-bit palettized, split_einsum_v2 attention (lower memory, Neural Engine)",
    },
    "float16-original": {
        "repo_id": "apple/coreml-stable-diffusion-2-1-base",
        "allow_patterns": ["original/compiled/*"],
        "compute_unit": "CPU_AND_GPU",
        "model_version": "stabilityai/stable-diffusion-2-1-base",
        "output_dir_name": "coreml-stable-diffusion",
        "description": "float16 original attention (highest quality, largest download)",
    },
    "float16-split-einsum": {
        "repo_id": "apple/coreml-stable-diffusion-2-1-base",
        "allow_patterns": ["split_einsum/compiled/*"],
        "compute_unit": "CPU_AND_NE",
        "model_version": "stabilityai/stable-diffusion-2-1-base",
        "output_dir_name": "coreml-stable-diffusion",
        "description": "float16 split_einsum attention (Neural Engine)",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hoist_single_child_dir(target_dir: Path) -> Path:
    entries = [path for path in target_dir.iterdir() if not path.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            destination = target_dir / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        nested.rmdir()
    return target_dir


def _download_zip_variant(variant: dict[str, str], output_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    archive_path = Path(
        hf_hub_download(
            repo_id=variant["repo_id"],
            filename=variant["zip_name"],
            token=token,
        )
    )
    extract_dir = output_dir / "Resources"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)

    nested_resources = extract_dir / "Resources"
    if nested_resources.is_dir():
        for item in nested_resources.iterdir():
            target = extract_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        nested_resources.rmdir()

    return _hoist_single_child_dir(extract_dir)


def _download_snapshot_variant(variant: dict[str, str], output_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    staging_dir = output_dir / "_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    downloaded = Path(
        snapshot_download(
            repo_id=variant["repo_id"],
            allow_patterns=variant["allow_patterns"],
            local_dir=staging_dir,
            token=token,
        )
    )

    compiled_root = next(downloaded.glob("*/compiled"))
    resources_dir = output_dir / "Resources"
    if resources_dir.exists():
        shutil.rmtree(resources_dir)
    shutil.move(str(compiled_root), str(resources_dir))
    shutil.rmtree(staging_dir, ignore_errors=True)
    return resources_dir


def _verify_resources(resources_dir: Path) -> None:
    required = ["TextEncoder.mlmodelc", "VAEDecoder.mlmodelc", "vocab.json", "merges.txt"]
    unet_options = ["Unet.mlmodelc", "UnetChunk1.mlmodelc"]
    missing = [name for name in required if not (resources_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Download incomplete; missing: {', '.join(missing)}")
    if not any((resources_dir / name).exists() for name in unet_options):
        raise RuntimeError("Download incomplete; missing Unet.mlmodelc or UnetChunk1.mlmodelc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="urpm-v13-split-einsum",
        help="Core ML model variant to download (default: urpm-v13-split-einsum)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where Resources/ will be created",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (falls back to HF_TOKEN env var)",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("Warning: Core ML image generation only runs on macOS.", file=sys.stderr)

    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print(
            "huggingface-hub is required. Install with: scripts/install.sh runtime",
            file=sys.stderr,
        )
        return 1

    variant = VARIANTS[args.variant]
    default_output = _project_root() / "models" / variant.get("output_dir_name", "coreml-stable-diffusion")
    output_dir = Path(args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.variant}: {variant['description']}")
    print(f"Repository: {variant['repo_id']}")

    if "zip_name" in variant:
        resources_dir = _download_zip_variant(variant, output_dir, args.token)
    else:
        resources_dir = _download_snapshot_variant(variant, output_dir, args.token)

    _verify_resources(resources_dir)

    print(f"\nCore ML assets ready at: {resources_dir}")
    print("\nAdd these lines to .env on the deployment host:")
    print(f"GENERATE_IMAGE_MODEL={variant['model_version']}")
    print(f"IMAGE_COREML_PATH={resources_dir}")
    print(f"IMAGE_COMPUTE_UNIT={variant['compute_unit']}")
    print("IMAGE_NUM_INFERENCE_STEPS=20")
    print("IMAGE_GUIDANCE_SCALE=7.5")
    print("\nThen restart the service: scripts/start.sh --background")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
