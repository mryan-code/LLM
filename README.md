# Storm Zero LLM

Storm Zero LLM is a local Python HTTP server that runs GGUF models with a layered training-memory architecture. Global rules and user data are loaded from MySQL at request time, intent is classified with a detection model, and prompts are routed to the appropriate generation path.

See `flow.md` for a step-by-step request flow summary.

## Running The Server

Install dependencies:

```bash
scripts/install.sh
```

On macOS, `scripts/install.sh` builds `llama-cpp-python` with Metal support and installs Core ML Stable Diffusion support. It also downloads the Core ML model if `IMAGE_COREML_PATH` is set in `.env` but the weights are missing.

Reinstall `llama-cpp-python` with Metal after `scripts/install.sh`:

```bash
scripts/install_llama_cpp.sh
```

Download or refresh Core ML weights manually:

```bash
scripts/download_coreml_sd.sh --variant urpm-v13-split-einsum
```

Other variants: `urpm-v13-original`, `palettized-original`, `float16-original`, `float16-split-einsum`. See `scripts/download_coreml_sd.py --help`.

Install only Core ML dependencies (after `scripts/install.sh`):

```bash
scripts/install_coreml.sh
```

Start or restart the server with PM2:

```bash
scripts/start.sh
```

Start in deploy/background compatibility mode:

```bash
scripts/start.sh --background
```

Sync global training from MySQL during start:

```bash
scripts/start.sh --seed-foundation
```

Stop the PM2-managed server:

```bash
scripts/stop.sh
```

You can also run the server in the foreground through the package CLI:

```bash
storm-zero-llm --host 127.0.0.1 --port 8765
```

Or use Python directly:

```bash
python -m storm_zero_llm --host 127.0.0.1 --port 8765
```

Sync global training from MySQL while starting:

```bash
python -m storm_zero_llm --seed-foundation
```

## Project Layout

```text
src/storm_zero_llm/
  agent.py        Request orchestration (/chat and /use-model flow)
  config.py       .env and runtime configuration
  detection.py    Intent classification (DETECTION_MODEL + heuristic fallback)
  memory.py       Deprecated file-backed memory store (unused by /chat)
  provider.py     llama-cpp-python GGUF provider, thinking strip, and local fallback
  reasoning.py    Response confidence and critique wrapping
  image_generation.py  Core ML Stable Diffusion and Flux image backends
  server.py       HTTP API
  training_db.py  MySQL global rules, user data, and conversation persistence
  tts.py          Kokoro text-to-speech
scripts/
  install.sh              Create venv and install package
  install_llama_cpp.sh    Install llama-cpp-python (Metal on macOS)
  install_coreml.sh       Install ml-stable-diffusion on macOS (Python 3.13-safe)
  ensure_coreml_model.sh  Download Core ML weights when IMAGE_COREML_PATH is missing
  download_coreml_sd.sh   Download Core ML Stable Diffusion model variants
  start.sh                Start or restart PM2 app
  stop.sh                 Stop PM2 app
tests/            pytest suite
```

## HTTP API

The local server uses Python's standard `http.server` stack and exposes a JSON API.

### GET /

Returns service metadata and endpoint list.

```json
{
  "status": "ok",
  "service": "storm-zero-llm",
  "message": "Storm Zero LLM server is running.",
  "readme": "# Storm Zero LLM ...",
  "endpoints": {
    "health": "/health",
    "chat": "/chat",
    "about": "/about",
    "use-model": "/use-model"
  }
}
```

### GET /health

Returns:

```json
{
  "success": true,
  "status_code": 200,
  "service": "storm-zero-llm",
  "readme": "{README.md contents}"
}
```

### GET /about

Returns `README.md` as markdown text.

### Request Flow (/chat and /use-model)

Both endpoints follow the same high-level runtime flow.

1. Load global rules and user data from MySQL and build the structured `prompts` response object.
2. If `user_data` is true, skip detection and go straight to user-data create/update/delete.
3. Otherwise detect query intent using the detection model.
4. Route the request to the appropriate generation path.
5. Attach media output when media is generated.
6. Strip model thinking from `response` and optionally return it in `reasoning`.
7. Optionally attach TTS output when `tts` is true.

#### 1) Global Rules and User Data Loading

Global rules are loaded from MySQL table `tblglobal_rule`:

- hard rules: `tblglobal_rule.strict = 1` AND `tblglobal_rule.deleted = 0`
  - cannot be overridden
- guidelines: `tblglobal_rule.strict = 0` AND `tblglobal_rule.deleted = 0`
  - can be overridden by user data

Each global hard-rule or guideline row is split on newlines (`\n`) into individual entries in `prompts.global_hard_rules.rules` / `prompts.global_guidelines.rules`.

User data is loaded by `user_id`, which must be a positive integer referencing `tbluser.id`:

- p2
  - user profile preferences stored as key/value pairs
  - loaded from `tbluser_p2` where `tbluser_p2.user_id` matches `parameter.user_id` and `tbluser_p2.deleted = 0`
  - returned in `prompts.user_p2.p2` as `- key = value` entries
- guidelines
  - user rules that can override global guidelines
  - loaded from `tbluser_guideline` where `tbluser_guideline.user_id` matches `parameter.user_id` and `tbluser_guideline.deleted = 0`
- avatar
  - loaded from `tbluser_avatar` for persona, names, and pronouns
  - returned in `prompts.avatar_data`

Request context is loaded from MySQL only. Local file-memory sync is unused by `/chat`.

Debug logging:

- The server prints loaded rule/user-data counts and input parameters when `parameter.env.GLOBAL_DEBUG_LEVEL == "debug"` or `parameter.env.DEBUG_USER == "mryan"`.
- Debug flags come from the request body's `env` object, not from server `.env`.

Example:

```json
{
  "user_id": 1,
  "prompt": "hello",
  "env": {
    "GLOBAL_DEBUG_LEVEL": "debug"
  }
}
```

#### 2) Query Detection

When `parameter.user_data` is true, detection is skipped and the request goes straight to user-data handling.

When `parameter.subject_id` is set to an existing subject, detection is skipped and the request goes straight to the base prompt with `conversation_mode=continued` and `subject_summary` set to that subject.

Otherwise detection builds a structured prompt object (`user_prompt`, `detection.txt` content, conversation subjects) and classifies intent. `DETECTION_MODEL` is used, or `POWER_DETECTION_MODEL` when the request sets `power` to true. If the detection model file is unavailable, the server falls back to heuristic classification. Referential prompts that match an existing subject continue that subject and return its `subject_id`.

Possible actions:

- create media (image or music)
- user data (p2 or guideline)
- scheduled task (for example reminders)
- general query

For general queries, the server determines whether the prompt starts a new conversation subject or continues an existing one:

- Subject text is normalized (lowercase words, no special characters).
- New conversation:
  - create a row in `tbluser_conversation_subject`
  - set `tbluser_conversation_subject.subject` to a brief summary of the prompt
  - set `tbluser_conversation_subject.user_id` = `parameters.user_id`
  - pass the created `id` as `subject_id`
- Continued conversation:
  - reuse existing `tbluser_conversation_subject.id` as `subject_id`
  - update `tbluser_conversation_subject.subject` when needed

Debug logging:

- Detection/classification output is printed when `parameter.env.GLOBAL_DEBUG_LEVEL == "debug"` or `parameter.env.DEBUG_USER == "mryan"`.

#### 3) Route to the Generation Path

All rules and user data from MySQL are passed to the selected model path. File-memory retrieval is not used.

- create media
  - image -> Core ML Stable Diffusion when `IMAGE_COREML_PATH` is set (macOS), otherwise Flux via `GENERATE_IMAGE_MODEL`
  - image prompt = `image_generation.txt` + user prompt + last 20 conversation turns
  - music -> `MUSIC_MODEL`
- user data
  - p2 -> upsert into `tbluser_p2` as `key`/`value` pairs extracted by the detection model
  - guideline -> create, or return existing `user_guideline_id` and ask how to proceed; update/delete with `user_data_id` + `user_data_action`
- scheduled task
  - handled by `BASE_MODEL` with scheduling context
- general query
  - route to `BASE_MODEL`, or `POWER_BASE_MODEL` when `power` is true
  - write a conversation row in `tbluser_conversation_content`:
    - `tbluser_conversation_content.user_conversation_subject_id` = `subject_id`
    - `tbluser_conversation_content.user_id` = `parameters.user_id`
    - `tbluser_conversation_content.prompt` = `parameters.prompt`
    - `tbluser_conversation_content.response` = final generated response (thinking stripped)

When `power` is true, detection uses `POWER_DETECTION_MODEL` instead of `DETECTION_MODEL`.

#### 4) Model Output Normalization

Qwen thinking models may emit a chain-of-thought block before the final answer (for example a `Thinking Process:` section or content inside model thinking tags).

By default, the server:

- disables model thinking during generation
- strips thinking blocks from model output
- returns only the final answer in `response`
- persists only the final answer to MySQL

When `include_reasoning` is `true`:

- thinking mode is enabled for that request
- `response` still contains only the final answer
- `reasoning` contains the extracted thinking text when the model produced any

`critique` remains the agent confidence note. It is not used for model thinking output.

For `/use-model`, steps 1 and 2 are the same. If detection resolves to create media, the media model is used. Otherwise the explicitly requested model path is used:

- `type=system`: `DEDICATED_MODELS_PATH/model`
- `type=user`: `CUSTOM_MODELS_PATH/model`

#### 5) Media Output

If media is generated, the response includes `media`:

```json
{
  "type": "media",
  "response": "",
  "media": {
    "base64": "...",
    "mime_type": "image/png"
  }
}
```

Use `media.base64` and `media.mime_type` for binary payloads in clients. Image responses leave `response` empty once media is attached.

Image generation on Apple Silicon uses [Apple ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) with pre-converted Core ML weights (for example [coreml-community/coreml-URPM-v13](https://huggingface.co/coreml-community/coreml-URPM-v13)). Set `IMAGE_COREML_PATH` to the compiled `Resources` directory containing `TextEncoder.mlmodelc`, `Unet.mlmodelc`, and `VAEDecoder.mlmodelc`. Set `GENERATE_IMAGE_MODEL` to the matching Hugging Face base model id used for tokenizer/scheduler setup (for example `runwayml/stable-diffusion-v1-5` for SD 1.5 Core ML packs).

Without `IMAGE_COREML_PATH`, image requests use Flux through diffusers (`GENERATE_IMAGE_MODEL`, optional `GENERATE_IMAGE_LORA_MODEL`).

#### 6) TTS Output

If `parameters.tts == true`, the server generates TTS:

- voice is always `parameters.voice` when provided
- default voice is `af_heart` when no voice is provided

TTS output shape:

```json
{
  "tts": {
    "base64": "...",
    "mime_type": "...",
    "voice": "..."
  }
}
```

TTS runs only when `tts` is explicitly `true`.

### POST /chat

Request body:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `user_id` | integer | yes | `tbluser.id` for rules and persistence |
| `prompt` | string | yes | User prompt |
| `topics` | string array | no | Optional subject hints for new conversations |
| `proactive_allowed` | boolean | no | Allows response to otherwise low-signal input |
| `include_reasoning` | boolean | no | When `true`, enables model thinking and adds a top-level `reasoning` field with the extracted thinking text |
| `power` | boolean | no | When `true`, uses `POWER_BASE_MODEL` and `POWER_DETECTION_MODEL` instead of the default models |
| `user_data` | boolean | no | When `true`, skip detection and handle user-data create/update/delete directly |
| `user_data_target` | string | no | `p2` or `guideline` (default `p2`) when `user_data` is true or detection routes to user data |
| `user_data_id` | integer | no | Existing `tbluser_guideline.id` for update/delete |
| `user_data_action` | string | no | `update` or `delete` when `user_data_id` is set; otherwise inferred from the prompt |
| `subject_id` | integer | no | When set to an existing subject, skip detection and continue that subject via the base prompt |
| `tts` | boolean | no | Generate TTS when `true` |
| `voice` | string | no | TTS voice; defaults to `af_heart` |
| `env` | object | no | Per-request debug flags (`GLOBAL_DEBUG_LEVEL`, `DEBUG_USER`) |

Example:

```bash
curl -s http://127.0.0.1:8765/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "prompt": "What do you remember about movies?",
    "topics": ["movies"],
    "tts": false
  }'
```

Response shape:

```json
{
  "success": true,
  "status_code": 200,
  "type": "conversation",
  "subject_id": 42,
  "response": "I hear you.",
  "confidence": "normal",
  "critique": "Response generated from conversation context.",
  "queries": [
    {
      "sql": "INSERT INTO tbluser_conversation_content (user_id, user_conversation_subject_id, prompt, response) VALUES (%s, %s, %s, %s)",
      "params": [1, 42, "how are you today", "I hear you."]
    }
  ],
  "prompts": {
    "user_prompt": "how are you today",
    "global_hard_rules": {
      "rules": ["Never respond like a program, or that you are incapable of emotion."],
      "description": "Always listen to Hard Rules..."
    },
    "global_guidelines": {
      "rules": ["Decide if you want to love or hate someone based on how they act to you over time the way a human would decide."],
      "description": "This is the global guidelines..."
    },
    "user_guidelines": {
      "rules": ["Refer to me as Sir"],
      "description": "This is the user's guidelines..."
    },
    "user_p2": {
      "p2": ["- favourite_music = I like rock"],
      "description": "This is the user's profile information..."
    },
    "avatar_data": {
      "description": "This is the avatar data...",
      "data": {
        "user_name": "Matt",
        "avatar_name": "Victoria",
        "avatar_persona": ["- emotional = 30%, rational = 70%"],
        "user_pronouns": "He/Him"
      }
    },
    "conversation_subjects": ["movies"],
    "conversation_subjects_content": [
      {
        "created": "2026-08-04 10:00:00",
        "prompt": "how are you today",
        "response": "I hear you."
      }
    ],
    "final": {
      "base": {
        "user_prompt": "how are you today",
        "global_hard_rules": ["Never respond like a program, or that you are incapable of emotion."],
        "global_guidelines": ["Decide if you want to love or hate someone based on how they act to you over time the way a human would decide."],
        "user_guidelines": ["Refer to me as Sir"],
        "user_p2": ["- favourite_music = I like rock"],
        "avatar_data": {
          "user_name": "Matt",
          "avatar_name": "Victoria",
          "avatar_persona": ["- emotional = 30%, rational = 70%"],
          "user_pronouns": "He/Him"
        },
        "conversation_subjects_content": [
          {
            "created": "2026-08-04 10:00:00",
            "prompt": "how are you today",
            "response": "I hear you."
          }
        ]
      },
      "image_generation": "",
      "detection": {
        "user_prompt": "how are you today",
        "content": "Classify the user prompt intent...",
        "conversation_subjects": ["movies"]
      }
    }
  },
  "rules_evaluated": {
    "global": true,
    "user": false
  }
}
```

`queries` contains every SQL statement executed for the request against MySQL, in run order.

`prompts` is the structured runtime context for the request: rules, p2, avatar data, and recent conversation rows loaded from MySQL, plus description text from `prompts/*.txt`.

`prompts.final` holds the prompts prepared for each stage:
- `base`: structured object used for generation (`user_prompt`, rules, p2, avatar, conversation content). For `conversation_mode=new`, content includes all rows for `user_id`; for `continued`, content is scoped to `subject_id` + `user_id`.
- `detection`: structured object (`user_prompt`, `content` from `prompts/detection.txt`, `conversation_subjects`).
- `image_generation`: composed prompt string.
- Unused stages are empty (`{}` / `""`).

If the prompt references an existing subject (for example "remember that time we talked about the weather"), detection continues that subject and returns its `subject_id`. Passing `subject_id` on the request skips detection and goes straight to the base prompt in continued mode.

When a guideline already exists, the response includes top-level `user_guideline_id` so the client can call again with `user_data`, `user_data_target=guideline`, `user_data_id`, and `user_data_action` (`update` or `delete`).

Possible `type` values:

- `conversation`
- `media`
- `user_data`
- `scheduled_task`

Notes:

- `/chat` uses the shared request flow above.
- `response` always contains the final model answer only; thinking is never included there.
- When `include_reasoning` is `true` and the model produced thinking text, the response also includes `reasoning`.
- Media responses include top-level `media.base64` and `media.mime_type`. Image responses leave `response` empty.
- If model output contains multiple explicit options (for example `Option 1`, `Option 2`), response text is normalized to `response.potential_options`.

Example with reasoning enabled:

```bash
curl -s http://127.0.0.1:8765/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "prompt": "this is a test",
    "include_reasoning": true
  }'
```

```json
{
  "success": true,
  "status_code": 200,
  "type": "conversation",
  "response": "Hey, test received.",
  "reasoning": "Thinking Process:\n\n1. Analyze the Request:\n...",
  "confidence": "normal",
  "critique": "Response generated from conversation context."
}
```

### POST /use-model

Request fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | string | yes | One of `system` or `user` |
| `user_id` | integer | yes | `tbluser.id` for rules and persistence |
| `model` | string | yes | Model filename relative to the selected models path |
| `prompt` | string | yes | Prompt to send to the model |
| `include_reasoning` | boolean | no | When `true`, enables model thinking and adds a top-level `reasoning` field with the extracted thinking text |
| `power` | boolean | no | When `true`, uses `POWER_DETECTION_MODEL` for intent detection (generation still uses the requested `model`) |
| `tts` | boolean | no | Generate TTS when `true` |
| `voice` | string | no | TTS voice; defaults to `af_heart` |
| `env` | object | no | Per-request debug flags (`GLOBAL_DEBUG_LEVEL`, `DEBUG_USER`) |

Model path resolution:

- `type=system`: `DEDICATED_MODELS_PATH/model`
- `type=user`: `CUSTOM_MODELS_PATH/model`

Example:

```bash
curl -s http://127.0.0.1:8765/use-model \
  -H "Content-Type: application/json" \
  -d '{
    "type": "system",
    "user_id": 1,
    "model": "text.gguf",
    "prompt": "Hello"
  }'
```

Response shape:

```json
{
  "success": true,
  "status_code": 200,
  "type": "system",
  "model": "/path/to/dedicated_models/text.gguf",
  "prompt": "Hello",
  "response": "I hear you.",
  "confidence": "normal",
  "critique": "Direct model invocation.",
  "rules_evaluated": {
    "global": true,
    "user": false
  }
}
```

Notes:

- `/use-model` uses the shared request flow above.
- `response` always contains the final model answer only.
- When `include_reasoning` is `true` and the model produced thinking text, the response also includes `reasoning`.
- If detection resolves to create media, the media model overrides the requested model path.
- Media responses include top-level `media.base64` and `media.mime_type`.
- If model output contains multiple explicit options, response text is normalized to `response.potential_options`.
- Model runtime failures return HTTP 502 with `error_code: "model_runtime_error"`.

## Training Data Structure

Global and user training are layered and isolated.

Global training is loaded from MySQL table `tblglobal_rule`:

- hard rules: `strict = 1` AND `deleted = 0`
- guidelines: `strict = 0` AND `deleted = 0`

User data is loaded at request time from:

- `tbluser_p2`
- `tbluser_guideline`

User conversation data is written to MySQL:

- `tbluser_conversation_subject` for subject grouping
- `tbluser_conversation_content` for prompt/response persistence

Guideline duplicates return the existing `user_guideline_id` instead of inserting again. Explicit update/delete uses `user_data_id` with `user_data_action`. See `foundation.md` for product-direction details on memory layering.

## Environment Variables

Server configuration is loaded from `.env` and process environment.

| Variable | Purpose |
| --- | --- |
| `LLM_HOST` | Server bind host |
| `LLM_PORT` | Server bind port |
| `PM2_APP` | PM2 process name |
| `CUSTOM_MODELS_PATH` | User model directory |
| `DEDICATED_MODELS_PATH` | System model directory |
| `BASE_MODEL` | Default conversation model |
| `DETECTION_MODEL` | Intent classification model |
| `POWER_BASE_MODEL` | Conversation model used when request `power` is true |
| `POWER_DETECTION_MODEL` | Detection model used when request `power` is true |
| `GENERATE_IMAGE_MODEL` | Hugging Face model id for image generation (Core ML reference pipeline or Flux model) |
| `GENERATE_IMAGE_LORA_MODEL` | LoRA weights for Flux image generation (ignored when `IMAGE_COREML_PATH` is set) |
| `IMAGE_COREML_PATH` | Path to compiled Core ML Resources directory; enables Core ML backend on macOS |
| `IMAGE_COMPUTE_UNIT` | Core ML compute unit (`CPU_AND_GPU`, `CPU_AND_NE`, etc.); default `CPU_AND_GPU` |
| `IMAGE_NUM_INFERENCE_STEPS` | Core ML / diffusion step count; default `20` |
| `IMAGE_GUIDANCE_SCALE` | Classifier-free guidance scale; default `7.5` |
| `IMAGE_DEVICE` | Flux backend device (`mps`, `cuda`, `cpu`) when Core ML is not used |
| `IMAGE_CPU_OFFLOAD` | Flux MPS CPU offload; default `true` |
| `IMAGE_SEQUENTIAL_CPU_OFFLOAD` | Flux sequential CPU offload on MPS; default `false` |
| `HG_TOKEN` / `HF_TOKEN` | Hugging Face token for model downloads |
| `MUSIC_MODEL` | Music generation model |
| `VIDEO_MODEL` | Video generation model |
| `TTS_PATH` | Path to Kokoro assets |
| `MEMORY_ROOT` | Deprecated local file memory root (unused by `/chat`) |
| `DB_USER`, `DB_PASS`, `DB_NAME`, `DB_HOST`, `DB_PORT` | MySQL connection |
| `LLM_N_GPU_LAYERS`, `LLM_N_THREADS` | llama-cpp runtime settings |
| `LLM_N_CTX` | Context window size in tokens; default `8192`. Use `0` only on hosts with enough memory for the model's full trained context |
| `LLM_N_BATCH` | Prompt-eval batch size for llama-cpp; default `512`. Larger values can speed long prompt processing on Metal |
| `LLM_FLASH_ATTN` | Enable flash attention in llama-cpp (`true`/`false`); default `true` |

All runtime model paths are env-driven. No model paths are hardcoded.

Example Core ML image setup (macOS):

```env
GENERATE_IMAGE_MODEL=runwayml/stable-diffusion-v1-5
IMAGE_COREML_PATH=/Users/inkt/Documents/Personal/storm_zero/llm/models/coreml-urpm-v13/Resources
IMAGE_COMPUTE_UNIT=CPU_AND_NE
IMAGE_NUM_INFERENCE_STEPS=20
IMAGE_GUIDANCE_SCALE=7.5
HF_TOKEN=your_huggingface_token
```

Per-request debug flags are not server env vars. Pass them in the request body's `env` object:

- `parameter.env.GLOBAL_DEBUG_LEVEL == "debug"`
- `parameter.env.DEBUG_USER == "mryan"`

## Testing

Run:

```bash
pytest
```

Current tests cover:

- `.env` config loading
- intent detection heuristics
- memory path naming and retrieval
- MySQL global training loader and conversation persistence
- agent orchestration and rule evaluation
- Core ML and Flux image generation routing
- model output normalization and optional reasoning extraction
- HTTP root, health, chat, and use-model endpoints

## Deployment

Deployment is defined in `.github/workflows/deploy.yml` and uses:

- `scripts/install.sh` (installs runtime deps, Core ML support on macOS, and re-downloads missing Core ML weights)
- `scripts/start.sh --background`
- `scripts/stop.sh`

Deploy rsync preserves `.env`, `.venv`, `models/`, and `data/` on the host. Core ML weights live under `models/` and are not in git.

Deployment files for the target host are also available at:

`smb://Matt's Mac mini._smb._tcp.local/inkt/Documents/Personal/storm_zero/llm`
