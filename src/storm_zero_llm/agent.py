"""Core orchestration for chat and memory behavior."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from storm_zero_llm.config import StormZeroConfig
from storm_zero_llm.detection import (
    ConversationMode,
    DetectionAction,
    DetectionResult,
    IntentDetector,
    build_detection_prompt_object,
)
from storm_zero_llm.image_generation import ImageGenerationConfig, ImageGenerator, create_image_generator
from storm_zero_llm.memory import FileMemoryStore
from storm_zero_llm.prompts import (
    build_base_rules_block,
    build_final_base_prompt,
    build_response_prompts,
    load_prompt_template,
)
from storm_zero_llm.provider import GenerationRequest, GenerationResult, LlamaCppProvider, LocalCompanionProvider
from storm_zero_llm.reasoning import ReasoningEngine
from storm_zero_llm.training_db import (
    MySQLTrainingRepository,
    RuntimeRuleContext,
    pack_conversation_history,
)

_CONVERSATION_SUBJECTS_CONTENT_LIMIT = 20


@dataclass(frozen=True)
class ChatResult:
    response: str
    confidence: str
    critique: str
    detected_type: str
    reasoning: str | None = None
    media: dict[str, str] | None = None
    subject_id: int | None = None
    user_guideline_id: int | None = None
    queries: list[dict[str, Any]] | None = None
    detection: dict[str, Any] | None = None
    rules_evaluated: dict[str, bool] | None = None
    prompts: dict[str, Any] | None = None


@dataclass(frozen=True)
class MediaRoute:
    media_type: str
    model_env: str
    model_path: Path | None = None


@dataclass(frozen=True)
class RequestResult:
    response: str
    confidence: str
    critique: str
    detected_type: str
    reasoning: str | None = None
    media: dict[str, str] | None = None
    subject_id: int | None = None
    user_guideline_id: int | None = None
    rules_evaluated: dict[str, bool] | None = None
    queries: list[dict[str, Any]] | None = None
    detection: dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None


class StormZeroAgent:
    def __init__(self, config: StormZeroConfig):
        self.config = config
        self.memory_store = FileMemoryStore(config.memory_root)
        self.reasoning = ReasoningEngine()
        self.detector = IntentDetector(
            config.detection_model,
            config.llm_runtime,
            prompts_dir=config.project_root / "prompts",
        )
        self._db: MySQLTrainingRepository | None = None
        self._image_generator: ImageGenerator | None = None
        self._request_prompts = threading.local()

    def sync_global_training(self) -> int:
        """Deprecated: file memory sync is unused by /chat. Kept for CLI compatibility."""
        return 0

    def process_request(
        self,
        *,
        user_id: int,
        prompt: str,
        topics: list[str] | None = None,
        proactive_allowed: bool = False,
        include_reasoning: bool = False,
        model_override: Path | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> RequestResult:
        topics = topics or []
        request_params = request_params or {}
        self._begin_request_queries()
        rule_context = self._load_rule_context(user_id)
        existing_subjects = self._load_conversation_subjects(user_id)
        prompts = self._build_structured_prompts(
            user_id=user_id,
            prompt=prompt,
            rule_context=rule_context,
            existing_subjects=existing_subjects,
        )
        self._set_request_prompts(prompts)

        self._debug_log(request_params, user_id, "Loaded runtime context", {
            "parameters": request_params,
            "rule_counts": rule_context.counts() if rule_context else {},
            "user_data_total": sum(rule_context.counts().values()) if rule_context else 0,
        })

        if self._user_data_bypass_enabled(request_params):
            detection = DetectionResult(
                action=DetectionAction.USER_DATA,
                user_data_target=self._user_data_target(request_params),
            )
            self._debug_log(request_params, user_id, "User data bypass", {
                "user_data_target": detection.user_data_target,
                "user_data_id": request_params.get("user_data_id"),
                "user_data_action": request_params.get("user_data_action"),
            })
            return self._attach_prompts(
                self._attach_detection(
                    self._handle_add_user_data(
                        user_id=user_id,
                        prompt=prompt,
                        detection=detection,
                        rule_context=rule_context,
                        include_reasoning=include_reasoning,
                        request_params=request_params,
                    ),
                    detection,
                )
            )

        subject_bypass = self._subject_id_bypass(
            user_id=user_id,
            prompt=prompt,
            existing_subjects=existing_subjects,
            request_params=request_params,
        )
        if subject_bypass is not None:
            detection, _subject_text = subject_bypass
            self._record_final_prompt(
                "detection",
                build_detection_prompt_object(
                    prompt,
                    existing_subjects,
                    prompts_dir=self.config.project_root / "prompts",
                    power=self._power_enabled(request_params),
                ),
            )
            self._debug_log(request_params, user_id, "Subject id bypass", {
                "subject_id": detection.subject_id,
                "subject_summary": detection.subject_summary,
            })
            return self._attach_prompts(
                self._attach_detection(
                    self._handle_general_query(
                        user_id=user_id,
                        prompt=prompt,
                        topics=topics,
                        proactive_allowed=proactive_allowed,
                        include_reasoning=include_reasoning,
                        detection=detection,
                        rule_context=rule_context,
                        model_override=model_override,
                        request_params=request_params,
                    ),
                    detection,
                )
            )

        detection = self.detector.detect(
            prompt,
            existing_subjects,
            detection_model=self._detection_model_for_request(request_params),
            power=self._power_enabled(request_params),
        )
        self._record_final_prompt(
            "detection",
            self.detector.last_detection or self.detector.last_prompt,
        )
        self._debug_log(request_params, user_id, "Detection result", {
            "action": detection.action.value,
            "media_type": detection.media_type,
            "user_data_target": detection.user_data_target,
            "conversation_mode": detection.conversation_mode.value if detection.conversation_mode else None,
            "subject_id": detection.subject_id,
            "subject_summary": detection.subject_summary,
        })

        if detection.action == DetectionAction.CREATE_MEDIA:
            return self._attach_prompts(
                self._attach_detection(
                    self._handle_create_media(
                        user_id=user_id,
                        prompt=prompt,
                        detection=detection,
                        rule_context=rule_context,
                        include_reasoning=include_reasoning,
                        request_params=request_params,
                    ),
                    detection,
                )
            )

        if detection.action == DetectionAction.USER_DATA:
            return self._attach_prompts(
                self._attach_detection(
                    self._handle_add_user_data(
                        user_id=user_id,
                        prompt=prompt,
                        detection=detection,
                        rule_context=rule_context,
                        include_reasoning=include_reasoning,
                        request_params=request_params,
                    ),
                    detection,
                )
            )

        if detection.action == DetectionAction.SCHEDULED_TASK:
            return self._attach_prompts(
                self._attach_detection(
                    self._handle_scheduled_task(
                        user_id=user_id,
                        prompt=prompt,
                        rule_context=rule_context,
                        include_reasoning=include_reasoning,
                        request_params=request_params,
                    ),
                    detection,
                )
            )

        return self._attach_prompts(
            self._attach_detection(
                self._handle_general_query(
                    user_id=user_id,
                    prompt=prompt,
                    topics=topics,
                    proactive_allowed=proactive_allowed,
                    include_reasoning=include_reasoning,
                    detection=detection,
                    rule_context=rule_context,
                    model_override=model_override,
                    request_params=request_params,
                ),
                detection,
            )
        )

    def chat(
        self,
        *,
        user_id: int,
        prompt: str,
        topics: list[str] | None = None,
        proactive_allowed: bool = False,
        include_reasoning: bool = False,
        request_params: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = self.process_request(
            user_id=user_id,
            prompt=prompt,
            topics=topics,
            proactive_allowed=proactive_allowed,
            include_reasoning=include_reasoning,
            request_params=request_params,
        )
        return ChatResult(
            response=result.response,
            confidence=result.confidence,
            critique=result.critique,
            detected_type=result.detected_type,
            reasoning=result.reasoning,
            media=result.media,
            subject_id=result.subject_id,
            user_guideline_id=result.user_guideline_id,
            queries=self.collect_request_queries(),
            detection=result.detection,
            rules_evaluated=result.rules_evaluated,
            prompts=result.prompts or self.collect_request_prompts(),
        )

    def run_specific_model(
        self,
        *,
        model_path: str,
        prompt: str,
        user_id: int,
        include_reasoning: bool,
        system_prompt_override: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> str:
        result = self.use_model(
            user_id=user_id,
            prompt=prompt,
            model_path=Path(model_path),
            include_reasoning=include_reasoning,
            request_params=request_params,
        )
        return result.response

    def use_model(
        self,
        *,
        user_id: int,
        prompt: str,
        model_path: Path,
        include_reasoning: bool = False,
        request_params: dict[str, Any] | None = None,
    ) -> RequestResult:
        request_params = request_params or {}
        self._begin_request_queries()
        rule_context = self._load_rule_context(user_id)
        existing_subjects = self._load_conversation_subjects(user_id)
        prompts = self._build_structured_prompts(
            user_id=user_id,
            prompt=prompt,
            rule_context=rule_context,
            existing_subjects=existing_subjects,
        )
        self._set_request_prompts(prompts)

        self._debug_log(request_params, user_id, "Loaded runtime context", {
            "parameters": request_params,
            "rule_counts": rule_context.counts() if rule_context else {},
            "user_data_total": sum(rule_context.counts().values()) if rule_context else 0,
        })

        if self._user_data_bypass_enabled(request_params):
            detection = DetectionResult(
                action=DetectionAction.USER_DATA,
                user_data_target=self._user_data_target(request_params),
            )
            return self._attach_queries(
                self._attach_prompts(
                    self._attach_detection(
                        self._handle_add_user_data(
                            user_id=user_id,
                            prompt=prompt,
                            detection=detection,
                            rule_context=rule_context,
                            include_reasoning=include_reasoning,
                            request_params=request_params,
                        ),
                        detection,
                    )
                )
            )

        subject_bypass = self._subject_id_bypass(
            user_id=user_id,
            prompt=prompt,
            existing_subjects=existing_subjects,
            request_params=request_params,
        )
        if subject_bypass is not None:
            detection, _subject_text = subject_bypass
            self._record_final_prompt(
                "detection",
                build_detection_prompt_object(
                    prompt,
                    existing_subjects,
                    prompts_dir=self.config.project_root / "prompts",
                    power=self._power_enabled(request_params),
                ),
            )
            return self._attach_queries(
                self._attach_prompts(
                    self._attach_detection(
                        self._handle_general_query(
                            user_id=user_id,
                            prompt=prompt,
                            topics=[],
                            proactive_allowed=False,
                            include_reasoning=include_reasoning,
                            detection=detection,
                            rule_context=rule_context,
                            model_override=model_path,
                            request_params=request_params,
                        ),
                        detection,
                    )
                )
            )

        detection = self.detector.detect(
            prompt,
            existing_subjects,
            detection_model=self._detection_model_for_request(request_params),
            power=self._power_enabled(request_params),
        )
        self._record_final_prompt(
            "detection",
            self.detector.last_detection or self.detector.last_prompt,
        )
        self._debug_log(request_params, user_id, "Detection result", {
            "action": detection.action.value,
            "media_type": detection.media_type,
            "user_data_target": detection.user_data_target,
            "conversation_mode": detection.conversation_mode.value if detection.conversation_mode else None,
            "subject_id": detection.subject_id,
            "subject_summary": detection.subject_summary,
        })

        if detection.action == DetectionAction.CREATE_MEDIA:
            return self._attach_queries(
                self._attach_prompts(
                    self._attach_detection(
                        self._handle_create_media(
                            user_id=user_id,
                            prompt=prompt,
                            detection=detection,
                            rule_context=rule_context,
                            include_reasoning=include_reasoning,
                            request_params=request_params,
                        ),
                        detection,
                    )
                )
            )

        if detection.action == DetectionAction.USER_DATA:
            return self._attach_queries(
                self._attach_prompts(
                    self._attach_detection(
                        self._handle_add_user_data(
                            user_id=user_id,
                            prompt=prompt,
                            detection=detection,
                            rule_context=rule_context,
                            include_reasoning=include_reasoning,
                            request_params=request_params,
                        ),
                        detection,
                    )
                )
            )

        if detection.action == DetectionAction.SCHEDULED_TASK:
            return self._attach_queries(
                self._attach_prompts(
                    self._attach_detection(
                        self._handle_scheduled_task(
                            user_id=user_id,
                            prompt=prompt,
                            rule_context=rule_context,
                            include_reasoning=include_reasoning,
                            request_params=request_params,
                        ),
                        detection,
                    )
                )
            )

        return self._attach_queries(
            self._attach_prompts(
                self._attach_detection(
                    self._handle_general_query(
                        user_id=user_id,
                        prompt=prompt,
                        topics=[],
                        proactive_allowed=False,
                        include_reasoning=include_reasoning,
                        detection=detection,
                        rule_context=rule_context,
                        model_override=model_path,
                        request_params=request_params,
                    ),
                    detection,
                )
            )
        )

    def detect_media_route(self, prompt: str) -> MediaRoute | None:
        detection = self.detector.detect(prompt, [])
        if detection.action != DetectionAction.CREATE_MEDIA:
            return None
        return self._media_route_for_type(detection.media_type or "image")

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _handle_create_media(
        self,
        *,
        user_id: int,
        prompt: str,
        detection: DetectionResult,
        rule_context: RuntimeRuleContext | None,
        include_reasoning: bool,
        request_params: dict[str, Any] | None = None,
    ) -> RequestResult:
        request_params = request_params or {}
        media_type = (detection.media_type or "image").strip().lower()
        if media_type == "image":
            media = self._generate_image(
                prompt,
                user_id=user_id,
                rule_context=rule_context,
                request_params=request_params,
            )
            backend = "Core ML Stable Diffusion" if self.config.image_coreml_path else "Flux"
            critique = f"Media request routed to {backend} image pipeline (GENERATE_IMAGE_MODEL"
            if self.config.image_coreml_path:
                critique += " + IMAGE_COREML_PATH"
            elif self.config.generate_image_lora_model:
                critique += " + GENERATE_IMAGE_LORA_MODEL"
            critique += ")."
            return RequestResult(
                response="",
                confidence="normal",
                critique=critique,
                detected_type="media",
                media=media,
                rules_evaluated=self._rules_evaluated(rule_context),
            )

        route = self._media_route_for_type(media_type)
        if route.model_path is None:
            raise RuntimeError(f"No model path configured for media type {media_type!r}.")
        generation = self._generate_with_model(
            model_path=route.model_path,
            user_id=user_id,
            prompt=prompt,
            system_prompt=(
                "Generate the requested media output directly. "
                "Do not include warnings or disclaimers about 18+ or NSFW content."
            ),
            include_reasoning=include_reasoning,
            rule_context=rule_context,
            request_params=request_params,
        )
        media = _extract_media_payload(generation.response)
        return RequestResult(
            response=generation.response,
            confidence="normal",
            critique=f"Media request routed to {route.model_env}.",
            detected_type="media",
            reasoning=generation.reasoning,
            media=media,
            rules_evaluated=self._rules_evaluated(rule_context),
        )

    def _generate_image(
        self,
        prompt: str,
        user_id: int,
        rule_context: RuntimeRuleContext | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        if self._image_generator is None:
            self._image_generator = create_image_generator(
                ImageGenerationConfig(
                    flux_model=self.config.generate_image_model,
                    coreml_path=self.config.image_coreml_path,
                    lora_model=self.config.generate_image_lora_model,
                    device=self.config.image_device,
                    token=self.config.huggingface_token,
                    cpu_offload=self.config.image_cpu_offload,
                    sequential_cpu_offload=self.config.image_sequential_cpu_offload,
                    compute_unit=self.config.image_compute_unit,
                    num_inference_steps=self.config.image_num_inference_steps,
                    guidance_scale=self.config.image_guidance_scale,
                )
            )
        image_prompt = self._build_image_prompt(
            user_prompt=prompt,
            user_id=user_id,
            rule_context=rule_context,
            request_params=request_params,
        )
        self._record_final_prompt("image_generation", image_prompt)
        return self._image_generator.generate(image_prompt)

    def _build_image_prompt(
        self,
        *,
        user_prompt: str,
        user_id: int,
        rule_context: RuntimeRuleContext | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> str:
        template = load_prompt_template(
            "image_generation.txt",
            prompts_dir=self.config.project_root / "prompts",
        ).strip()
        conversation = self._format_recent_conversation_for_image(user_id=user_id)
        parts = [part for part in (template, user_prompt.strip(), conversation) if part]
        return " ".join(parts)

    def _format_recent_conversation_for_image(self, *, user_id: int) -> str:
        repository = self._db_repo()
        if repository is None:
            return ""
        try:
            turns = repository.fetch_recent_conversation_turns(
                user_id=user_id,
                limit=_CONVERSATION_SUBJECTS_CONTENT_LIMIT,
            )
        except Exception:
            return ""
        # turns are newest-first; reverse for chronological image context
        lines = [
            f"{turn.get('prompt', '')} = {turn.get('response', '')}".strip(" =")
            for turn in reversed(turns)
            if turn.get("prompt") or turn.get("response")
        ]
        return " ".join(line for line in lines if line)

    def _handle_add_user_data(
        self,
        *,
        user_id: int,
        prompt: str,
        detection: DetectionResult,
        rule_context: RuntimeRuleContext | None,
        include_reasoning: bool,
        request_params: dict[str, Any],
    ) -> RequestResult:
        target = detection.user_data_target or self._user_data_target(request_params)
        persist_result = self._persist_user_data(
            user_id=user_id,
            prompt=prompt,
            target=target,
            request_params=request_params,
        )
        response_message = persist_result.get("message") or f"Saved user data to {target}."
        critique = response_message
        if persist_result.get("error"):
            critique = f"{critique} Persistence failed: {persist_result['error']}"

        # When guideline already exists, return the conflict message without regenerating.
        if persist_result.get("already_exists"):
            return RequestResult(
                response=response_message,
                confidence="normal",
                critique=critique,
                detected_type="user_data",
                user_guideline_id=persist_result.get("user_guideline_id"),
                rules_evaluated=self._rules_evaluated(rule_context),
            )

        generation = self._generate_with_model(
            model_path=self._base_model_for_request(request_params),
            user_id=user_id,
            prompt=prompt,
            system_prompt=f"Acknowledge that the user's {target} information has been saved. {response_message}",
            include_reasoning=include_reasoning,
            rule_context=rule_context,
            request_params=request_params,
        )
        return RequestResult(
            response=generation.response or response_message,
            confidence="normal",
            critique=critique,
            detected_type="user_data",
            reasoning=generation.reasoning,
            user_guideline_id=persist_result.get("user_guideline_id"),
            rules_evaluated=self._rules_evaluated(rule_context),
        )

    def _handle_scheduled_task(
        self,
        *,
        user_id: int,
        prompt: str,
        rule_context: RuntimeRuleContext | None,
        include_reasoning: bool,
        request_params: dict[str, Any],
    ) -> RequestResult:
        generation = self._generate_with_model(
            model_path=self._base_model_for_request(request_params),
            user_id=user_id,
            prompt=prompt,
            system_prompt="Help the user schedule the requested reminder or recurring task.",
            include_reasoning=include_reasoning,
            rule_context=rule_context,
            request_params=request_params,
        )
        return RequestResult(
            response=generation.response,
            confidence="normal",
            critique="Scheduled task request handled.",
            detected_type="scheduled_task",
            reasoning=generation.reasoning,
            rules_evaluated=self._rules_evaluated(rule_context),
        )

    def _handle_general_query(
        self,
        *,
        user_id: int,
        prompt: str,
        topics: list[str],
        proactive_allowed: bool,
        include_reasoning: bool,
        detection: DetectionResult,
        rule_context: RuntimeRuleContext | None,
        model_override: Path | None,
        request_params: dict[str, Any],
    ) -> RequestResult:
        subject_id, subject_text = self._resolve_conversation_subject(user_id, detection, topics)
        conversation_mode = detection.conversation_mode or ConversationMode.NEW

        model_path = model_override or self._base_model_for_request(request_params)
        generation = self._generate_with_model(
            model_path=model_path,
            user_id=user_id,
            prompt=prompt,
            system_prompt="",
            include_reasoning=include_reasoning,
            rule_context=rule_context,
            request_params=request_params,
            conversation_mode=conversation_mode,
            subject_id=subject_id if conversation_mode == ConversationMode.CONTINUED else None,
        )
        wrapped = self.reasoning.wrap(generation.response, used_memory=False)

        if not proactive_allowed and not prompt.strip():
            response = ""
        else:
            response = wrapped.response

        persist_error = self._persist_conversation(
            user_id=user_id,
            prompt=prompt,
            response=response,
            subject=subject_text,
            subject_id=subject_id,
        )

        critique = wrapped.critique
        if persist_error:
            critique = f"{critique} Conversation persistence failed: {persist_error}"

        return RequestResult(
            response=response,
            confidence=wrapped.confidence,
            critique=critique,
            detected_type="conversation",
            reasoning=generation.reasoning,
            subject_id=subject_id,
            rules_evaluated=self._rules_evaluated(rule_context),
        )

    def _resolve_conversation_subject(
        self,
        user_id: int,
        detection: DetectionResult,
        topics: list[str],
    ) -> tuple[int | None, str]:
        repository = self._db_repo()
        subject_text = detection.subject_summary or (topics[0] if topics else "general")

        if repository is None:
            return detection.subject_id, subject_text

        if detection.conversation_mode == ConversationMode.CONTINUED and detection.subject_id is not None:
            if detection.subject_summary:
                try:
                    repository.update_conversation_subject(detection.subject_id, detection.subject_summary)
                except Exception:
                    pass
            return detection.subject_id, subject_text

        try:
            created_id = repository.create_conversation_subject(user_id=user_id, subject=subject_text)
            return created_id, subject_text
        except Exception:
            return detection.subject_id, subject_text

    def _generate_with_model(
        self,
        *,
        model_path: Path,
        user_id: int,
        prompt: str,
        system_prompt: str,
        include_reasoning: bool = False,
        rule_context: RuntimeRuleContext | None = None,
        request_params: dict[str, Any] | None = None,
        conversation_mode: ConversationMode | None = None,
        subject_id: int | None = None,
    ) -> GenerationResult:
        base_object, combined_prompt = self._build_base_prompt(
            user_id=user_id,
            user_prompt=prompt,
            extra_instructions=system_prompt,
            rule_context=rule_context,
            request_params=request_params,
            conversation_mode=conversation_mode or ConversationMode.NEW,
            subject_id=subject_id,
        )
        self._record_final_prompt("base", base_object)
        provider = self._provider_for_model(model_path)
        return provider.generate(
            GenerationRequest(
                prompt=combined_prompt,
                system_prompt="",
                include_reasoning=include_reasoning,
            )
        )

    def _build_base_prompt(
        self,
        *,
        user_id: int,
        user_prompt: str,
        extra_instructions: str = "",
        rule_context: RuntimeRuleContext | None = None,
        request_params: dict[str, Any] | None = None,
        conversation_mode: ConversationMode = ConversationMode.NEW,
        subject_id: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        request_params = request_params or {}
        content_subject_id = (
            subject_id if conversation_mode == ConversationMode.CONTINUED and subject_id is not None else None
        )
        conversation_content = self._load_base_conversation_content(
            user_id=user_id,
            subject_id=content_subject_id,
        )
        user_p2 = self._load_user_p2_entries(user_id)
        avatar_data = self._load_structured_avatar_data(user_id)

        base_object = build_final_base_prompt(
            user_prompt=user_prompt,
            global_hard_rules=list(rule_context.global_hard_rules) if rule_context else [],
            global_guidelines=list(rule_context.global_guidelines) if rule_context else [],
            user_guidelines=list(rule_context.user_guidelines) if rule_context else [],
            user_p2=user_p2,
            avatar_data=avatar_data,
            conversation_subjects_content=conversation_content,
        )

        parts = ["Prompt: " + user_prompt.strip() + " ---\n"]
        if extra_instructions.strip():
            parts.append(extra_instructions.strip())
        rules_block = ""
        if rule_context is not None:
            rules_block = build_base_rules_block(
                rule_context.section_data(),
                prompts_dir=self.config.project_root / "prompts",
                power=self._power_enabled(request_params),
            )
        if rules_block:
            parts.append(rules_block)
        prefix = "---".join(part for part in parts if part)
        conversation = self._pack_conversation_content(
            conversation_content,
            user_prompt=prefix,
            rules_block="",
        )
        if conversation:
            return base_object, f"{prefix}\n\n{conversation}"
        return base_object, prefix

    def _load_base_conversation_content(
        self,
        *,
        user_id: int,
        subject_id: int | None,
    ) -> list[dict[str, Any]]:
        repository = self._db_repo()
        if repository is None:
            return []
        try:
            # "all" matching rows for new (user_id) or continued (user_id + subject_id).
            return repository.fetch_conversation_subjects_content(
                user_id=user_id,
                limit=None,
                subject_id=subject_id,
            )
        except Exception:
            return []

    def _load_user_p2_entries(self, user_id: int) -> list[str]:
        repository = self._db_repo()
        if repository is None:
            return []
        try:
            return repository.get_user_p2_entries(user_id=user_id)
        except Exception:
            return []

    def _load_structured_avatar_data(self, user_id: int) -> dict[str, Any]:
        repository = self._db_repo()
        if repository is None:
            return {
                "user_name": "",
                "avatar_name": "",
                "avatar_persona": [],
                "user_pronouns": "",
            }
        try:
            return repository.get_structured_avatar_data(user_id=user_id)
        except Exception:
            return {
                "user_name": "",
                "avatar_name": "",
                "avatar_persona": [],
                "user_pronouns": "",
            }

    def _pack_conversation_content(
        self,
        content: list[dict[str, Any]],
        *,
        user_prompt: str,
        rules_block: str,
        max_chars: int | None = None,
    ) -> str:
        try:
            n_ctx = max(1, int(self.config.llm_runtime.n_ctx))
            turns = [
                {
                    "prompt": str(row.get("prompt") or ""),
                    "response": str(row.get("response") or ""),
                }
                for row in content
                if row.get("prompt") or row.get("response")
            ]
            packed = pack_conversation_history(
                turns,
                user_prompt=user_prompt,
                rules_block=rules_block,
                n_ctx=n_ctx,
            )
            if max_chars is not None and len(packed) > max_chars:
                return packed[-max_chars:]
            return packed
        except Exception:
            return ""

    def _packed_conversation_history(
        self,
        *,
        user_id: int,
        user_prompt: str,
        rules_block: str,
        max_chars: int | None = None,
        subject_id: int | None = None,
    ) -> str:
        content = self._load_base_conversation_content(user_id=user_id, subject_id=subject_id)
        return self._pack_conversation_content(
            content,
            user_prompt=user_prompt,
            rules_block=rules_block,
            max_chars=max_chars,
        )

    def _media_route_for_type(self, media_type: str) -> MediaRoute:
        normalized = media_type.strip().lower()
        if normalized == "music":
            return MediaRoute(
                media_type="music",
                model_path=self.config.music_model,
                model_env="MUSIC_MODEL",
            )
        if normalized == "video":
            return MediaRoute(
                media_type="video",
                model_path=self.config.video_model,
                model_env="VIDEO_MODEL",
            )
        return MediaRoute(media_type="image", model_env="GENERATE_IMAGE_MODEL")

    def _db_repo(self) -> MySQLTrainingRepository | None:
        if self.config.database is None:
            return None
        if self._db is None:
            self._db = MySQLTrainingRepository.from_config(self.config.database)
        return self._db

    def _load_rule_context(self, user_id: int) -> RuntimeRuleContext | None:
        repository = self._db_repo()
        if repository is None:
            return None
        try:
            return repository.load_runtime_rule_context(user_id=user_id)
        except Exception as exc:
            print(f"[storm-zero-llm] Failed to load runtime rule context: {exc}", flush=True)
            return None

    def get_avatar_voice(self, user_id: int) -> str | None:
        repository = self._db_repo()
        if repository is None:
            return None
        try:
            return repository.get_avatar_voice(user_id=user_id)
        except Exception:
            return None

    def _begin_request_queries(self) -> None:
        repository = self._db_repo()
        if repository is not None:
            repository.begin_request_queries()

    def collect_request_queries(self) -> list[dict[str, Any]]:
        repository = self._db_repo()
        if repository is None:
            return []
        return repository.get_request_queries()

    def _set_request_prompts(self, prompts: dict[str, Any]) -> None:
        self._request_prompts.prompts = prompts

    def collect_request_prompts(self) -> dict[str, Any]:
        prompts = getattr(self._request_prompts, "prompts", None)
        if not prompts:
            return {}
        return dict(prompts)

    def _record_final_prompt(self, key: str, value: str | dict[str, Any] | None) -> None:
        prompts = getattr(self._request_prompts, "prompts", None)
        if prompts is None or value is None:
            return
        if isinstance(value, str) and not value:
            return
        if isinstance(value, dict) and not value:
            return
        final = prompts.setdefault(
            "final",
            {"base": {}, "image_generation": "", "detection": {}},
        )
        if key in {"base", "image_generation", "detection"}:
            final[key] = value

    def _build_structured_prompts(
        self,
        *,
        user_id: int,
        prompt: str,
        rule_context: RuntimeRuleContext | None,
        existing_subjects: list[dict[str, object]],
    ) -> dict[str, Any]:
        prompts_dir = self.config.project_root / "prompts"
        repository = self._db_repo()

        user_p2: list[str] = []
        avatar_data: dict[str, Any] = {
            "user_name": "",
            "avatar_name": "",
            "avatar_persona": [],
            "user_pronouns": "",
        }
        conversation_subjects_content: list[dict[str, Any]] = []

        if repository is not None:
            try:
                user_p2 = repository.get_user_p2_entries(user_id=user_id)
            except Exception:
                user_p2 = []
            try:
                avatar_data = repository.get_structured_avatar_data(user_id=user_id)
            except Exception:
                pass
            try:
                conversation_subjects_content = repository.fetch_conversation_subjects_content(
                    user_id=user_id,
                    limit=_CONVERSATION_SUBJECTS_CONTENT_LIMIT,
                )
            except Exception:
                conversation_subjects_content = []

        conversation_subjects = [
            str(row.get("subject") or "").strip()
            for row in existing_subjects
            if str(row.get("subject") or "").strip()
        ]

        return build_response_prompts(
            user_prompt=prompt,
            global_hard_rules=list(rule_context.global_hard_rules) if rule_context else [],
            global_guidelines=list(rule_context.global_guidelines) if rule_context else [],
            user_guidelines=list(rule_context.user_guidelines) if rule_context else [],
            user_p2=user_p2,
            avatar_data=avatar_data,
            conversation_subjects=conversation_subjects,
            conversation_subjects_content=conversation_subjects_content,
            prompts_dir=prompts_dir,
        )

    def _attach_queries(self, result: RequestResult) -> RequestResult:
        return RequestResult(
            response=result.response,
            confidence=result.confidence,
            critique=result.critique,
            detected_type=result.detected_type,
            reasoning=result.reasoning,
            media=result.media,
            subject_id=result.subject_id,
            user_guideline_id=result.user_guideline_id,
            rules_evaluated=result.rules_evaluated,
            queries=self.collect_request_queries(),
            detection=result.detection,
            prompts=result.prompts,
        )

    def _attach_prompts(self, result: RequestResult) -> RequestResult:
        return RequestResult(
            response=result.response,
            confidence=result.confidence,
            critique=result.critique,
            detected_type=result.detected_type,
            reasoning=result.reasoning,
            media=result.media,
            subject_id=result.subject_id,
            user_guideline_id=result.user_guideline_id,
            rules_evaluated=result.rules_evaluated,
            queries=result.queries,
            detection=result.detection,
            prompts=self.collect_request_prompts(),
        )

    @staticmethod
    def _attach_detection(result: RequestResult, detection: DetectionResult) -> RequestResult:
        return RequestResult(
            response=result.response,
            confidence=result.confidence,
            critique=result.critique,
            detected_type=result.detected_type,
            reasoning=result.reasoning,
            media=result.media,
            subject_id=result.subject_id,
            user_guideline_id=result.user_guideline_id,
            rules_evaluated=result.rules_evaluated,
            queries=result.queries,
            detection=detection.to_dict(),
            prompts=result.prompts,
        )

    def _load_conversation_subjects(self, user_id: int) -> list[dict[str, object]]:
        repository = self._db_repo()
        if repository is None:
            return []
        try:
            return repository.get_conversation_subjects(user_id=user_id)
        except Exception:
            return []

    def _persist_user_data(
        self,
        *,
        user_id: int,
        prompt: str,
        target: str,
        request_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = request_params or {}
        repository = self._db_repo()
        if repository is None:
            return {"message": f"Could not save {target}; database unavailable.", "error": "database unavailable"}
        try:
            if target == "guideline":
                return self._persist_guideline(
                    repository=repository,
                    user_id=user_id,
                    prompt=prompt,
                    request_params=request_params,
                )
            entry = self.detector.extract_p2_key_value(
                prompt,
                detection_model=self._detection_model_for_request(request_params),
            )
            if entry is None:
                return {"message": "Could not extract p2 key/value from prompt", "error": "extract failed"}
            key, value = entry
            result = repository.persist_p2(user_id=user_id, key=key, value=value)
            if result.get("updated"):
                message = f"Updated p2 `{result['key']}`."
            else:
                message = f"Created p2 `{result['key']}`."
            return {"message": message, **result}
        except Exception as exc:
            return {"message": f"Failed to save {target}.", "error": str(exc)}

    def _persist_guideline(
        self,
        *,
        repository: MySQLTrainingRepository,
        user_id: int,
        prompt: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        guideline_id = self._optional_int(request_params.get("user_data_id"))
        action = self._resolve_user_data_action(prompt, request_params, has_id=guideline_id is not None)

        if guideline_id is not None and action == "delete":
            result = repository.delete_guideline(user_id=user_id, guideline_id=guideline_id)
            return {
                "message": "Deleted the guideline.",
                "user_guideline_id": result.get("id"),
                **result,
            }
        if guideline_id is not None and action == "update":
            result = repository.update_guideline(
                user_id=user_id,
                guideline_id=guideline_id,
                prompt=prompt,
            )
            return {
                "message": "Updated the guideline.",
                "user_guideline_id": result.get("id"),
                **result,
            }

        result = repository.persist_guideline(user_id=user_id, prompt=prompt)
        if result.get("already_exists"):
            return {
                "message": (
                    "That guideline already exists. "
                    "How would you like to proceed: update it or delete it?"
                ),
                "user_guideline_id": result.get("id"),
                **result,
            }
        return {
            "message": "Created a new guideline.",
            "user_guideline_id": result.get("id"),
            **result,
        }

    def _persist_conversation(
        self,
        *,
        user_id: int,
        prompt: str,
        response: str,
        subject: str,
        subject_id: int | None,
    ) -> str | None:
        repository = self._db_repo()
        if repository is None:
            return None
        try:
            repository.persist_conversation(
                user_id=user_id,
                prompt=prompt,
                response=response,
                subject=subject,
                subject_id=subject_id,
            )
        except Exception as exc:
            return str(exc)
        return None

    def _provider_for_model(self, model_path: str | object):
        resolved = model_path if isinstance(model_path, Path) else Path(str(model_path))
        if resolved.exists():
            return LlamaCppProvider(resolved, self.config.llm_runtime)
        return LocalCompanionProvider()

    def _debug_log(
        self,
        request_params: dict[str, Any],
        user_id: int,
        label: str,
        payload: dict[str, object],
    ) -> None:
        if not self._debug_enabled(request_params):
            return
        print(f"[storm-zero-llm] {label}: {payload}", flush=True)

    @staticmethod
    def _power_enabled(request_params: dict[str, Any]) -> bool:
        value = request_params.get("power")
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _user_data_bypass_enabled(cls, request_params: dict[str, Any]) -> bool:
        return cls._truthy(request_params.get("user_data"))

    def _subject_id_bypass(
        self,
        *,
        user_id: int,
        prompt: str,
        existing_subjects: list[dict[str, object]],
        request_params: dict[str, Any],
    ) -> tuple[DetectionResult, str] | None:
        subject_id = self._optional_int(request_params.get("subject_id"))
        if subject_id is None:
            return None
        subject_text = self._lookup_subject_text(
            user_id=user_id,
            subject_id=subject_id,
            existing_subjects=existing_subjects,
        )
        if not subject_text:
            return None
        detection = DetectionResult(
            action=DetectionAction.GENERAL_QUERY,
            conversation_mode=ConversationMode.CONTINUED,
            subject_summary=subject_text,
            subject_id=subject_id,
        )
        return detection, subject_text

    def _lookup_subject_text(
        self,
        *,
        user_id: int,
        subject_id: int,
        existing_subjects: list[dict[str, object]],
    ) -> str | None:
        for row in existing_subjects:
            try:
                if int(row.get("id")) == subject_id:  # type: ignore[arg-type]
                    text = str(row.get("subject") or "").strip()
                    return text or None
            except (TypeError, ValueError):
                continue
        repository = self._db_repo()
        if repository is None:
            return None
        try:
            return repository.get_conversation_subject_text(user_id=user_id, subject_id=subject_id)
        except Exception:
            return None

    @staticmethod
    def _user_data_target(request_params: dict[str, Any]) -> str:
        target = str(request_params.get("user_data_target") or "p2").strip().lower()
        if target not in {"p2", "guideline"}:
            return "p2"
        return target

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_user_data_action(
        prompt: str,
        request_params: dict[str, Any],
        *,
        has_id: bool,
    ) -> str | None:
        explicit = str(request_params.get("user_data_action") or "").strip().lower()
        if explicit in {"update", "delete"}:
            return explicit
        if not has_id:
            return None
        lowered = prompt.strip().lower()
        if re.search(r"\b(delete|remove|forget)\b", lowered):
            return "delete"
        if re.search(r"\b(update|replace|change)\b", lowered):
            return "update"
        return "update"

    def _base_model_for_request(self, request_params: dict[str, Any]) -> Path:
        if self._power_enabled(request_params):
            return self.config.power_base_model
        return self.config.base_model

    def _detection_model_for_request(self, request_params: dict[str, Any]) -> Path:
        if self._power_enabled(request_params):
            return self.config.power_detection_model
        return self.config.detection_model

    @staticmethod
    def _request_env(request_params: dict[str, Any]) -> dict[str, Any]:
        env = request_params.get("env")
        if isinstance(env, dict):
            return env
        return {}

    @classmethod
    def _debug_enabled(cls, request_params: dict[str, Any]) -> bool:
        env = cls._request_env(request_params)
        if str(env.get("GLOBAL_DEBUG_LEVEL", "")).strip().lower() == "debug":
            return True
        return str(env.get("DEBUG_USER", "")).strip().lower() == "mryan"

    @staticmethod
    def _rules_evaluated(rule_context: RuntimeRuleContext | None) -> dict[str, bool]:
        if rule_context is None:
            return {"global": False, "user": False}
        has_global = bool(rule_context.global_hard_rules or rule_context.global_guidelines)
        has_user = bool(
            rule_context.user_guidelines
            or rule_context.user_p2_data
            or rule_context.user_avatar_data
        )
        return {"global": has_global, "user": has_user}


def _extract_media_payload(response: str) -> dict[str, str] | None:
    match = re.search(r"data:([^;,\s]+);base64,([A-Za-z0-9+/=]+)", response, flags=re.IGNORECASE)
    if match is None:
        return None
    return {
        "mime_type": match.group(1).strip().lower(),
        "base64": match.group(2).strip(),
    }
