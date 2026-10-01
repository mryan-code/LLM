# Request Flow

This is the flow of a request to the `/chat` or `/use-model` endpoint.

See `README.md` for full API and environment documentation.

## 1. Global Rules and User Data Are Loaded

### Global rules

Loaded from MySQL table `tblglobal_rule`:

- hard rules: `tblglobal_rule.strict = 1` AND `tblglobal_rule.deleted = 0`
  - cannot be overridden
- guidelines: `tblglobal_rule.strict = 0` AND `tblglobal_rule.deleted = 0`
  - can be overridden by user data
- each row is split on `\n` into individual entries in the prompts rules arrays

### User data (p2, guidelines)

- p2
  - user information like name or favourite colour
    - stored in `tbluser_p2` where `tbluser_p2.user_id` matches `parameter.user_id` (`tbluser.id`) and `tbluser_p2.deleted = 0`
- guidelines
  - user rules that can override global guidelines
    - stored in `tbluser_guideline` where `tbluser_guideline.user_id` matches `parameter.user_id` (`tbluser.id`) and `tbluser_guideline.deleted = 0`

### Debug logging

Rule counts, user-data counts, and input parameters are printed when:

```text
parameter.env.GLOBAL_DEBUG_LEVEL == "debug"
or
parameter.env.DEBUG_USER == "mryan"
```

These values come from the request body's `env` object, not from server `.env`.

## 2. Structured Prompts Object

After loading MySQL context, the response `prompts` object is built:

- `user_prompt`
- `global_hard_rules` / `global_guidelines` / `user_guidelines` — `{ rules, description }`
- `user_p2` — `{ p2, description }`
- `avatar_data` — `{ description, data }`
- `conversation_subjects`
- `conversation_subjects_content` — newest 20 rows from `tbluser_conversation_content`
- `final` — prompts prepared for each stage:
  - `base`: structured generation object
    - `user_prompt`, `global_hard_rules`, `global_guidelines`, `user_guidelines`, `user_p2`, `avatar_data`, `conversation_subjects_content`
    - new conversation: all conversation content for `user_id`
    - continued conversation: conversation content for `user_id` + `subject_id`
  - `detection`: structured detection object
    - `user_prompt`, `content` from `prompts/detection.txt`, `conversation_subjects`
  - `image_generation`: image prompt string
  - unused stages are empty (`{}` / `""`)

Descriptions come from `prompts/*.txt` (`global_hard_rules.txt`, `global_guidelines.txt`, `user_guidelines.txt`, `user_p2.txt`, `avatar_data.txt`).

## 3. Query Detection

- If `parameter.user_data` is true, detection is skipped and the request goes straight to user-data handling with `user_data_target` / `user_data_id` / `user_data_action`.
- If `parameter.subject_id` is set to an existing subject, detection is skipped and the request goes straight to the base prompt with `conversation_mode=continued` and `subject_summary` set to that subject.
- Otherwise detection builds `prompts.final.detection` and classifies intent.
- If the prompt references an existing subject (for example "remember that time we talked about the weather"), continue that subject and return its `subject_id` with the response.
- `DETECTION_MODEL` is used when the model file is available.
- Heuristic fallback is used when the detection model is unavailable.

Possible actions:

- create media
  - image or music
  - a suitable response is generated
- user data
  - p2 or guideline
  - a suitable response is generated
- scheduled task
  - for example "remind me to take my medication at 5:00pm"
- general query
  - catch-all conversation path
  - determine whether this is a continued conversation or a completely new one
  - when generating `tbluser_conversation_subject.subject`, no special characters are used and all words are lowercase
  - new
    - create an entry in `tbluser_conversation_subject`
    - `tbluser_conversation_subject.subject` is a brief summary of the prompt
    - `tbluser_conversation_subject.user_id` is `parameters.user_id`
    - pass `tbluser_conversation_subject.id` as `subject_id` for step 4
  - continued
    - pass existing `tbluser_conversation_subject.id` as `subject_id` for step 4
    - update `tbluser_conversation_subject.subject` if needed

Detection output is printed when:

```text
parameter.env.GLOBAL_DEBUG_LEVEL == "debug"
or
parameter.env.DEBUG_USER == "mryan"
```

## 4. Route to the Generation Path

All rules and user data are passed to the selected model. File-memory retrieval is not used.

- create media
  - image -> `GENERATE_IMAGE_MODEL` (prompt = `image_generation.txt` + user prompt + last 20 conversation turns)
  - music -> `MUSIC_MODEL`
- user data
  - p2 -> upsert `tbluser_p2` (`key` <= 3 words, `value` <= 100 chars)
  - guideline -> create, or return existing `user_guideline_id` and ask how to proceed; update/delete when `user_data_id` + action are provided
- scheduled task
  - handled by `BASE_MODEL`
- general query
  - routed to `BASE_MODEL`
  - create an entry in `tbluser_conversation_content`
    - `tbluser_conversation_content.user_conversation_subject_id` = `subject_id` from step 3
    - `tbluser_conversation_content.user_id` = `parameters.user_id`
    - `tbluser_conversation_content.prompt` = `parameters.prompt`
    - `tbluser_conversation_content.response` = model response

For `/use-model`, steps 1 through 3 are the same. If detection resolves to create media, the media model is used. Otherwise the requested model path is used:

- `type=system` -> `DEDICATED_MODELS_PATH/model`
- `type=user` -> `CUSTOM_MODELS_PATH/model`

## 5. Media Output

If media is created, the response includes:

```json
{
  "media": {
    "base64": "...",
    "mime_type": "..."
  }
}
```

## 6. TTS Output

If `parameters.tts == true`, generate a TTS object:

- always use `parameters.voice` when provided
- default voice is `af_heart` when no voice is provided

```json
{
  "tts": {
    "base64": "...",
    "mime_type": "..."
  }
}
```

## 7. Queries Output

Both `/chat` and `/use-model` responses include a top-level `queries` array listing every SQL statement executed during the request, in run order.

```json
{
  "queries": [
    {
      "sql": "SELECT * FROM tblglobal_rule WHERE strict = %s AND deleted = 0",
      "params": [1]
    },
    {
      "sql": "INSERT INTO tbluser_conversation_content (user_id, user_conversation_subject_id, prompt, response) VALUES (%s, %s, %s, %s)",
      "params": [1, 42, "how are you today", "I am doing well."]
    }
  ]
}
```
