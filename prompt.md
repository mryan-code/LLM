# Historical Prompt Log

This file records implementation prompts over time. It is not the canonical API reference.

For current behavior, use:

- `README.md`
- `flow.md`
- `AGENTS.md`

---

# 2026-06-30

- Create a training endpoint that allows the user to train the model on a specific topic, or add hard rules to the model.
- Update the readme to reflect the new training endpoint.

# 2026-07-02

- Global training hard rules are stored in `tblglobal_rule` where `strict` = 1 AND `deleted` = 0
- Global training guidelines are stored in `tblglobal_rule` where `strict` = 0 AND `deleted` = 0
- update the code and documentation to reflect this 

# 2026-07-03 use existing models

- create a new endpoint that allows the user to use an existing model
- the parameters should be type, option, model and prompt
  - type should be one of: "system", "user"
    - system: an option of (text, nswf, image, music, code) is required, and model is optional
    - user: a model is required, and option is optional
  - prompt should be a string

- update the README to reflect the new endpoint

# 2026-07-03 update training endpoint

- remove all training logic and endpoints 

# 2026-07-03 rewright the whole LLM
- build the following app in python

## Running the Server

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

Install all dependancies:

```bash
scripts/install.sh
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

## HTTP API

The local server is implemented with Python's standard `http.server` stack and exposes a small JSON API.

### `GET /`

Returns service metadata:

```json
{
	"status": "ok",
	"service": "storm-zero-llm",
	"message": "Storm Zero LLM server is running.",
	"readme": "# Storm Zero ...",
	"endpoints": {
		"health": "/health",
		"chat": "/chat",
		"about": "/about",
		"use-model": "/use-model"
	}
}
```

### `GET /health`

Returns health status:

```json
{
  "success": true,
  "status_code": 200,
  "service": "storm-zero-llm",
  "readme": "{README.md}"
}
```

### `GET /about`

Returns the project `README.md` as markdown text.

### `POST /chat`

Sends a prompt to the agent:

```bash
curl -s http://127.0.0.1:8765/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "owner",
    "prompt": "What do you remember about movies?",
    "topics": ["movies"],
    "proactive_allowed": false
  }'
```

Request fields:

| Field               | Type         | Purpose                                                                            |
| ------------------- | ------------ | ---------------------------------------------------------------------------------- |
| `user_id`           | string       | Identity used for authority and memory visibility                                  |
| `prompt`            | string       | User prompt                                                                        |
| `topics`            | string array | Optional memory retrieval topics                                                   `memory_write`, `feature_control`, `app_control`, or `owner_admin` |
| `proactive_allowed` | boolean      | Allows the reasoning layer to respond to otherwise low-signal input                |

`include_reasoning` is optional; when set to `true`, `/chat` returns raw model or agent output.

Response shape:

```json
{
  "success": true,
  "status_code": 200,
  "response": "I hear you...",
  "confidence": "normal",
  "critique": "Response generated without matching long-term memory."
}
```

### `POST /use-model`

Uses an existing model with specified parameters:

```bash
curl -s http://127.0.0.1:8765/use-model \
  -H "Content-Type: application/json" \
  -d '{
    "type": "system",
    "option": "text",
    "prompt": "Hello, how are you?"
  }'
```

Request fields:

- `type` (string, required): `"system"` or `"user"`
- `option` (string, required when type is `"system"`): one of `"text"`, `"nsfw"`, `"image"`, `"music"`, `"code"` (`"nswf"` is accepted as a legacy alias)
- `model` (string, required when type is `"user"`; optional when type is `"system"`)
- `prompt` (string, required): the prompt to send to the model
- `include_reasoning` (boolean, optional): when `true`, returns raw model output including reasoning text; default is `false`

When `type` is `"system"`, the `DEDICATED_MODELS_PATH` environment variable is used to construct the model path as `DEDICATED_MODELS_PATH/option.gguf`. For example, if `DEDICATED_MODELS_PATH=/models` and `option=text`, the endpoint will look for `/models/{option}.gguf`.

When `type` is `"user"`, the `CUSTOM_MODELS_PATH` environment variable is used to construct the model path as `CUSTOM_MODELS_PATH/model`. For example, if `CUSTOM_MODELS_PATH=/custom-models` and `model=my-model.gguf`, the endpoint will look for `/custom-models/my-model.gguf`.

Response shape:

```json
{
  "success": true,
  "status_code": 200,
  "type": "system",
  "option": "text",
  "model": "/models/text.gguf",
  "prompt": "Hello, how are you?",
  "response": "Hello. I am doing well and ready to help."
}
```

## Training Data Structure

Training data should be organized as a layered memory system. The goal is for the AI to know the difference between rules it must obey, guidance it should generally follow, and contextual memories it can use to understand a person or conversation. The memory system should be readable, searchable, editable, and modular so new memory types can be added later without rewriting the whole LLM.

Training data should be split into global training data and user-based training data. Global data applies to the entire AI instance. User-based data applies to one specific user and should only influence that user's experience unless the owner explicitly promotes it to global memory.

- caveates:
  - global
    - hard rule (strict, always adhered to)
    - system guideline (can be overridden by a user guideline)
  - user
    - p2 (information about user, eg: name, age, favourite colour)
    - user guiideline (can overrid system guidelines, but not override hard rules)
    - conversation (default, grouped by subject)

### global training data
  The LLM startup path should open a MySQL connection in main and pull the executable global training layers from the database before serving requests. Global hard rules and global guidelines come from `tblglobal_rule`: hard rules use `strict = 1 AND deleted = 0`, and guidelines use `strict = 0 AND deleted = 0`. `foundation.md` remains the product-direction document that defines how those records should be modeled and interpreted.

### user-based training data
  Determine from the context, whether the input is user info(p2), user rule(guildeline), or general input(conversation).
  If it's p2 add it to `tbluser_p2` with the `user_id` column set to the user_id parameter and the data in the `p2` column.
  If it's guideline add it to `tbluser_guideline` with the `user_id` column set to the user_id parameter and the data in the `guideline` column.
  If it's conversation add it to `tbluser_conversation_content` with the `user_id` column set to the user_id parameter, the `user_conversation_subject_id`to the id of the subject, and the data in the `conversation` column, but first create a row in `tbluser_conversation_subject` with the `user_id` column set to the user_id parameter, if this conversation doesnt already have one.
  When the LLM is corrected mark the old data `deleted` = 1

## env vars
- server data
  - LLM_HOST = "127.0.0.1"
    - this is the host the LLM server starts on
  - LLM_PORT = "8765"
    - this is the port the LLM server starts on
  - PM2_APP = "sz-llm"
    - this is the name used for the pm2 instance

- models
  - CUSTOM_MODELS_PATH="/Users/inkt/Documents/Personal/Storm Zero/LLM/Storm-Zero-LLM/models"
  - DEDICATED_MODELS_PATH="/Users/inkt/Documents/Personal/Storm Zero/LLM/Storm-Zero-LLM/dedicated_models"
  - BASE_MODEL="/Users/inkt/Documents/Personal/Storm Zero/LLM/Storm-Zero-LLM/dedicated_models/base.gguf"

- database connection
  - DB_USER="dev"
  - DB_PASS='VenomousRodent510MYS!'
  - DB_NAME='dbstorm_zero'
  - DB_HOST='192.168.50.192'
  - DB_PORT='3307'

- various llm related values
  - LLM_N_GPU_LAYERS=-1
  - LLM_N_THREADS=8
  - LLM_N_CTX=1024

- always use env variables and add new ones if needed, never hardcode

## querying the LLM

- use  the model ```env.BASE_MODEL``` to run the query, but first apply the database sourced traing, outlined above

## misc
- documentation is maintained in `README.md`, `flow.md`, and `AGENTS.md`
- use `llama-cpp-python` for running any model

## Testing

Run the full test suite:

```bash
pytest
```

The tests cover:

- `.env` config loading
- local GGUF model discovery
- authority restrictions
- memory path naming and retrieval order
- user memory privacy
- agent fallback provider behavior
- onboarding question count
- HTTP root, health, chat, and train endpoints
- start and stop Python script helpers

## Deployment

Deployment is defined in `.github/workflows/deploy.yml` and runs on pushes to `main`.

The workflow:

1. Checks out the repository.
2. Creates an SSH deploy key from repository secrets.
3. Checks TCP reachability to the remote host.
4. Stops the currently running remote server if present.
5. Syncs repository files to the remote deploy path with `rsync`.
6. Runs `scripts/install.sh` on the remote host.
7. Starts the server with `scripts/start.sh --background --seed-foundation`.
8. Waits for `/health` to return successfully.

Important deployment inputs:

| Secret or variable | Purpose            |
| ------------------ | ------------------ |
| `DEPLOY_HOST`      | Remote host        |
| `DEPLOY_PORT`      | SSH port           |
| `DEPLOY_USERNAME`  | SSH username       |
| `DEPLOY_PATH`      | Remote app path    |
| `DEPLOY_SSH_KEY`   | SSH private key    |
| `SERVER_PORT`      | Remote server port |

The workflow preserves `.env`, excludes `.git`, `.github`, and local GGUF model files, and expects the remote host to have Python and `rsync` available.

# 2026-07-10 No more tokens or temp

- remove tokens(max_tokens) and temparature from being used everywhere, remove all reference to them in docs and env or variables

# 2026-07-14 Query detection

- when the chat emdpoint is hit, use the basic model ```env.BASE_MODEL``` to determine the type, if is p2, guideline or conversation
  - if it is a p2 (piece of user data), create an row in tbluser_p2 and return a message aknowledging
  - if it is a guideline (user rule or setting), and there isnt a strict system rule blocking it, add a row to tbluser_guideline, and return a message aknowleging it
  - else create a row in `tbluser_conversation_subject` if there isnt already a subject grouping, then create a row in `tbluser_conversation_content` with `user_conversation_subject_id` being the `id` of the suject that groups it, the `prompt` is the user's input, and the `response` is what the ```env.BASE_MODEL``` returns
- return the type as a parameter in the response

# 2026-07-14 Evaluate rules

- when the chat endpoint is hit, evaluate global and user rules, EVERYTIME
- user rule context includes `tbluser_guideline` and `tbluser_p2` with `deleted` = 0 for that `user_id`

# 2026-07-14 TTS
- add a text to speech service on both /chat and /use-model endpoints
- please use kokoro (https://github.com/hexgrad/kokoro)
- the path to the tts assests are located in ```env.TTS_PATH```
- only use the tts service if there is a parameter of tts = true passed in

# 2026-07-17 classify prompt for user rule
- rewrite the training data system and add back fetching the BASE_MODEL to configuration
- run the `parameter.prompt` against `env.BASE_MODEL` to determine if it's p2, a guideline or a general conversation
  - if it is a p2 (piece of user data), create an row in tbluser_p2 and return a message aknowledging
  - if it is a guideline (user rule or setting), and there isnt a strict system rule blocking it, add a row to tbluser_guideline, and return a message aknowleging it
  - else create a row in `tbluser_conversation_subject` if there isnt already a subject grouping, then create a row in `tbluser_conversation_content` with `user_conversation_subject_id` being the `id` of the suject that groups it, the `prompt` is the user's input, and the `response` is what the ```env.BASE_MODEL``` returns
- return the type as a parameter in the response

# 2026-07-17 return file
- if the return a file, return the full base64 string `data:...;base64,...` in a separate element in the return
- this can be either /chat or /use-model
- and it is not relative to TTS

# 2026-07-17 return file as oject separatinng the base64 and mime_type

- eg: file = {
  base64:...,
  mime:...
}

# 2026-07-19 genarate media models
- if the prompt is to create a media (image, music, video), use the following models
- do not warn about 18+ or NSFW content
- use `env.GENERATE_IMAGE_MODEL` to generate images
- use `env.MUSIC_MODEL` to generate music
- use `env.VIDEO_MODEL` to generate video
- return a relivant message
- make sure this applies /chat and /use-model endpoints

# 2026-07-19 remove avatar from the system

# 2026-07-19 remove all user training, no guidelines or p2, always a conversation suject/content, so no detection and no BASIC_MODEL

# 2026-07-25 p2
- user data p2 now has a key value format
- ```SELECT `key`, `value` FROM `tbluse_p2` WHERE `user_id` = parameter.user_id and `deleted` = 0;```
- when adding/updating userdata/p2 in the detection llm, add it to that table, it that format
- when fetching the p2 rules list them in the following format:
  "The following is user_data (P2), these are user preferences:
    - `key`=`value`
    - `key`=`value`
    - `key`=`value`"

# 2026-07-29 prompts
- when running a detection, use prompts/detection.txt as the prompt, prepend the parameters.prompt, and append f"\n{subjects_block}"
- when fetching subjects, query as many rows as the model permits (prompt\n+system prompt\n+subjects)

# 2026-07-29 base prompt
- when building a prompt for the base, start with the user's prompt append the rules (global hard + guidelines), and a breif description for each, user guidelines, p2, avatar data, and as many rows of conversation content as env.n_ctx allows
- use prompts/base.txt as an example
- no more writing to text files, everything will be pulled fron the db as needed
- the conversation query will be "SELECT `prompt`, `response` FROM `tbluser_conversation_content` WHERE `user_id` = %s AND `deleted` = 0 ORDER BY `created` DESC LIMIT %s;"
  - no more `created` col
  - this is an example of a a conversation row
    - eg: "- prompt = response+\n"

# 2026-07-29 image generation prompt
- when building a prompt for the base, start with the user's prompt append the p2, avatar data, and as many rows of conversation content as env.n_ctx allows
- use prompts/image_generation.txt as an example

# 2026-07-29 prompts general
- the files `prompts/*.txt` are constantly being updated use them as a template to guide generating the systam prompts on each request

# 2026-07-29 prompts general
- interprit the text contained in '[]' as instructions for generating the prompt in `prompts/*.txt`

# 2026-08-04 update prompts response
- the prompts object in the response is a object with the following properties:
  - user_prompt: the user's prompt `parameter.prompt`
  - global_hard_rules:
    - this should be populated with and object containing the following properties:
      - rules: an array of rules `SELECT `rule` FROM `tblglobal_rule` WHERE `strict` = 1 AND `deleted` = 0;` eg: Never respond like a program, or that you are incapable of emotion.
      - description: a description of the rules populated from `prompts/global_hard_rules.txt`
  - global_guidelines:
    - this should be populated with and object containing the following properties:
      - rules: an array of rules `SELECT `rule` FROM `tblglobal_rule` WHERE `strict` = 0 AND `deleted` = 0;` eg: - Decide if you want to love or hate someone based on how they act to you over time the way a human would decide.
      - description: a description of the rules populated from `prompts/global_guidelines.txt`
  - user_guidelines:
    - this should be populated with and object containing the following properties:
      - rules: an array of rules `SELECT `guideline` FROM `tbluser_guideline` WHERE `user_id` = parameter.user_id AND `deleted` = 0;` eg: - Refer to me as Sir
      - description: a description of the rules populated from `prompts/user_guidelines.txt`
  - user_p2:
    - this should be populated with and object containing the following properties:
      - p2: an array of p2 `SELECT `key`, `value` FROM `tbluser_p2` WHERE `user_id` = parameter.user_id AND `deleted` = 0;` eg: - favourite_music = I like rock
      - description: a description of the p2 populated from `prompts/user_p2.txt`
  - avatar_data:
    - this should be populated with and object containing the following properties:
      - description: a description of the avatar data populated from `prompts/avatar_data.txt`
      - data: an object containing the following properties:
        - user_name (This is what I refer to the user as): `user_name`
        - avatar_name (This is what I refer to myself(the LLM) as): `avatar_name`
        - avatar_persona(reply using these values, col starts with 'persona_'):
          - `persona_emotional_rational`= strip 'persona_', convert the remaining '_' to a space, the first term will be preceeded by '- ', followed by ' = "+(100 - `value`) + '%, ', the second term will be followed by '= " + `value` + '%' eg: - emotional = 30%, rational = 70%
        - user_pronouns(these are the pronouns I use to describe the user): `user_pronouns`
  - conversation_subjects: the conversation subjects `SELECT `subject` FROM `tbluser_conversation_subject` WHERE `user_id` = parameter.user_id AND `deleted` = 0;`
  - conversation_subjects_content: the conversation subjects content `SELECT `created`,`prompt`, `response` FROM `tbluser_conversation_content` WHERE `user_id` = parameter.user_id AND `deleted` = 0 ORDER BY `created` DESC LIMIT %s;`

  - if `parameter.user_data` is true, skip detection and go straight to user_data creation or update, a parameter for the target object will be passed in, it can be p2 or guideline, an id for the object will be passed in
  - when executing the detection llm, use user_prompt + " - " + `prompts/detection.txt` as the prompt
  - when executing the base llm, use the following:
    - create_media:
      - media_type:
        - image:
          - description: a description of the image generation populated from `prompts/image_generation.txt`
          - prompt: the prompt for the image generation populated from `prompts/image_generation.txt` + " " + user_prompt + " " + the most recent 20 rows of conversation content
          - return a media object with the following properties:
            - base64: the base64 string of the image
            - mime_type: the mime type of the image
    - user_data:
      - user_data_target:
        - p2:
          - create an entry in tbluser_p2 with the `user_id` and `key` and `value` being the key and value of the p2
            - the key is a short description of the p2, no more than 3 words eg: favourite_music
            - the value is the user's prompt, intelligently summarized to 100 characters or less
            - if the key already exists, update the value
          - return a message aknowledging the p2 creation or the update
        - guideline:
          - the guideline is the user's prompt, intelligently summarized to 100 characters or less, no special characters, except for spaces
          - if the guideline already exists, return a message saying it already exists, and ask the user how they want to proceed, return a user_guideline_id
              - if the user wants to update it, update the guideline and return a message aknowledging the update
              - if the user wants to delete it, delete the guideline and return a message aknowledging the deletion, and mark the deleted row as 1
          - else create an entry in tbluser_guideline with the `user_id` and `guideline` being the guideline and return a message aknowledging the new guideline
          - return a message aknowledging the guideline creation or the update
    - general_query:
      - subject_summary:
        - brief lowercase summary without special characters, this will be used to create the subject
      - conversation_mode:
        - new:
          - create an entry in tbluser_conversation_subject with the `user_id` and `subject` being the user's prompt
          - create an entry in tbluser_conversation_content with the `user_id`, `user_conversation_subject_id`, `prompt`, and `response` being the user's prompt and the response from the base llm
          - return a message aknowledging the new subject and the response from the base llm
        - continued:
          - create an entry in tbluser_conversation_content with the `user_id`, `user_conversation_subject_id`, `prompt`, and `response` being the user's prompt and the response from the base llm
          - update to be more specific
          - return a message aknowledging the continued subject and the response from the base llm


  - no more writing to text files, everything will be pulled fron the db as needed
  - update any docs for this change

# 2026-08-04 update prompts response
- include the final prompts object in the response:
  - "final": {
    - "base": the base prompt string
    - "image_generation": the image generation prompt string
    - "detection": the detection prompt string
  }

# 2026-08-04 update base prompt
- when building the the base prompt, use the following format:
  - new :
    - "base": {
      - "user_prompt": the user's prompt
      - "global_hard_rules": the global hard rules
      - "global_guidelines": the global guidelines
      - "user_guidelines": the user's guidelines
      - "user_p2": the user's p2
      - "avatar_data": the avatar data
      - "conversation_subjects_content": all the conversation subjects content for that user_id
    }
  - continued:
    - "base": {
      - "user_prompt": the user's prompt
      - "global_hard_rules": the global hard rules
      - "global_guidelines": the global guidelines
      - "user_guidelines": the user's guidelines
      - "user_p2": the user's p2
      - "avatar_data": the avatar data
      - "conversation_subjects_content": all the conversation subjects content for the subject_id passed in and the user_id
    }

# 2026-08-04 update detection prompt
- if the prompt references a specific subject, like "remember that time we talked about the weather", if the subject already exists, return the subject_id, and and appropriate message
- if a subject_id is passed in, pypass the detection llm and go straight to the base prompt, setting the conversation_mode to continued, and the subject_summary to the subject
- when building the the detection prompt, use the following format:
  - "detection": {
    - "user_prompt": the user's prompt
    - content: `prompts/detection.txt`
    - conversation_subjects: the conversation subjects `SELECT `subject` FROM `tbluser_conversation_subject` WHERE `user_id` = parameter.user_id AND `deleted` = 0;`
  }

# 2026-08-04 global rules prompts
- when building the global rules and global guidelines prompts, each data row should be split into an array at "\n", and the array should be added to the appropriate rules array in the prompts object