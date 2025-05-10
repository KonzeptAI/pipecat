#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from . import events
from .audio_transcriber import AudioTranscriber

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


def language_to_gemini_language(language: Language) -> Optional[str]:
    """Maps a Language enum value to a Gemini Live supported language code.

    Source:
    https://ai.google.dev/api/generate-content#MediaResolution

    Returns None if the language is not supported by Gemini Live.
    """
    language_map = {
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN_IN: "bn-IN",
        # Chinese (Mandarin)
        Language.CMN: "cmn-CN",
        Language.CMN_CN: "cmn-CN",
        Language.ZH: "cmn-CN",  # Map general Chinese to Mandarin for Gemini
        Language.ZH_CN: "cmn-CN",  # Map Simplified Chinese to Mandarin for Gemini
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # English
        Language.EN: "en-US",  # Default to US English (though not explicitly listed in supported codes)
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain Spanish
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # French
        Language.FR: "fr-FR",  # Default to France French
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese (Brazil)
        Language.PT_BR: "pt-BR",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }
    return language_map.get(language)


class GeminiMultimodalLiveContext(OpenAILLMContext):
    @staticmethod
    def upgrade(obj: OpenAILLMContext) -> "GeminiMultimodalLiveContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GeminiMultimodalLiveContext):
            logger.debug(f"Upgrading to Gemini Multimodal Live Context: {obj}")
            obj.__class__ = GeminiMultimodalLiveContext
            obj._restructure_from_openai_messages()
        return obj

    def _restructure_from_openai_messages(self):
        pass

    def extract_system_instructions(self):
        system_instruction = ""
        for item in self.messages:
            if item.get("role") == "system":
                content = item.get("content", "")
                if content:
                    if system_instruction and not system_instruction.endswith("\n"):
                        system_instruction += "\n"
                    system_instruction += str(content)
        return system_instruction

    def get_messages_for_initializing_history(self):
        messages = []
        for item in self.messages:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        return messages


class GeminiMultimodalLiveUserContextAggregator(OpenAIUserContextAggregator):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # kind of a hack just to pass the LLMMessagesAppendFrame through, but it's fine for now
        if isinstance(frame, LLMMessagesAppendFrame):
            await self.push_frame(frame, direction)


class GeminiMultimodalLiveAssistantContextAggregator(OpenAIAssistantContextAggregator):
    # The LLMAssistantContextAggregator uses TextFrames to aggregate the LLM output,
    # but the GeminiMultimodalLiveAssistantContextAggregator pushes LLMTextFrames and TTSTextFrames. We
    # need to override this proces_frame for LLMTextFrame, so that only the TTSTextFrames
    # are process. This ensures that the context gets only one set of messages.
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if not isinstance(frame, LLMTextFrame):
            await super().process_frame(frame, direction)

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        # We don't want to store any images in the context. Revisit this later
        # when the API evolves.
        pass


@dataclass
class GeminiMultimodalLiveContextAggregatorPair:
    _user: GeminiMultimodalLiveUserContextAggregator
    _assistant: GeminiMultimodalLiveAssistantContextAggregator

    def user(self) -> GeminiMultimodalLiveUserContextAggregator:
        return self._user

    def assistant(self) -> GeminiMultimodalLiveAssistantContextAggregator:
        return self._assistant


class GeminiMultimodalModalities(Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class GeminiMediaResolution(str, Enum):
    """Media resolution options for Gemini Multimodal Live."""

    UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"  # Use default
    LOW = "MEDIA_RESOLUTION_LOW"  # 64 tokens
    MEDIUM = "MEDIA_RESOLUTION_MEDIUM"  # 256 tokens
    HIGH = "MEDIA_RESOLUTION_HIGH"  # Zoomed reframing with 256 tokens


class GeminiVADParams(BaseModel):
    """Voice Activity Detection parameters."""

    disabled: Optional[bool] = Field(default=None)
    start_sensitivity: Optional[events.StartSensitivity] = Field(default=None)
    end_sensitivity: Optional[events.EndSensitivity] = Field(default=None)
    prefix_padding_ms: Optional[int] = Field(default=None)
    silence_duration_ms: Optional[int] = Field(default=None)


class ContextWindowCompressionParams(BaseModel):
    """Parameters for context window compression."""

    enabled: bool = Field(default=False)
    trigger_tokens: Optional[int] = Field(
        default=None
    )  # None = use default (80% of context window)


class InputParams(BaseModel):
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    modalities: Optional[GeminiMultimodalModalities] = Field(
        default=GeminiMultimodalModalities.AUDIO
    )
    language: Optional[Language] = Field(default=Language.EN_US)
    media_resolution: Optional[GeminiMediaResolution] = Field(
        default=GeminiMediaResolution.UNSPECIFIED
    )
    vad: Optional[GeminiVADParams] = Field(default=None)
    context_window_compression: Optional[ContextWindowCompressionParams] = Field(default=None)
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GeminiMultimodalLiveLLMService(LLMService):
    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        model="models/gemini-2.0-flash-live-001",
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        tools: Optional[Union[List[dict], ToolsSchema]] = None,
        transcribe_user_audio: bool = False,
        params: InputParams = InputParams(),
        inference_on_context_initialization: bool = True,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self._last_sent_time = 0
        self._api_key = api_key
        self._base_url = base_url
        self.set_model_name(model)
        self._voice_id = voice_id
        self._language_code = params.language

        self._system_instruction = system_instruction
        self._tools = tools
        self._inference_on_context_initialization = inference_on_context_initialization
        self._needs_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused
        self._context = None
        self._websocket = None
        self._receive_task = None
        self._transcribe_audio_task = None
        self._transcribe_audio_queue = asyncio.Queue()

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._transcriber = AudioTranscriber(api_key)
        self._transcribe_user_audio = transcribe_user_audio
        self._user_is_speaking = False
        self._bot_is_speaking = False
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""

        self._sample_rate = 24000

        self._language = params.language
        self._language_code = (
            language_to_gemini_language(params.language) if params.language else "en-US"
        )
        self._vad_params = params.vad

        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "max_tokens": params.max_tokens,
            "presence_penalty": params.presence_penalty,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "modalities": params.modalities,
            "language": self._language_code,
            "media_resolution": params.media_resolution,
            "vad": params.vad,
            "context_window_compression": params.context_window_compression.model_dump()
            if params.context_window_compression
            else {},
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        
        # Track outstanding tool calls and state
        self._outstanding_tool_ids: set[str] = set()
        self._is_initial_context = True
        self._waiting_for_model_turn_after_tool = False

    def can_generate_metrics(self) -> bool:
        return True

    def set_audio_input_paused(self, paused: bool):
        self._audio_input_paused = paused

    def set_video_input_paused(self, paused: bool):
        self._video_input_paused = paused

    def set_model_modalities(self, modalities: GeminiMultimodalModalities):
        self._settings["modalities"] = modalities

    def set_language(self, language: Language):
        """Set the language for generation."""
        self._language = language
        self._language_code = language_to_gemini_language(language) or "en-US"
        self._settings["language"] = self._language_code
        logger.info(f"Set Gemini language to: {self._language_code}")

    async def set_context(self, context: OpenAILLMContext):
        """Set the context explicitly from outside the pipeline.

        This is useful when initializing a conversation because in server-side VAD mode we might not have a
        way to trigger the pipeline. This sends the history to the server. The `inference_on_context_initialization`
        flag controls whether to set the turnComplete flag when we do this. Without that flag, the model will
        not respond. This is often what we want when setting the context at the beginning of a conversation.
        """
        if self._context:
            logger.error(
                "Context already set. Can only set up Gemini Multimodal Live context once."
            )
            return
        self._context = GeminiMultimodalLiveContext.upgrade(context)
        await self._create_initial_response()
        self._is_initial_context = False  # Mark that initial context has been processed

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        self._bot_is_speaking = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        self._waiting_for_model_turn_after_tool = False  # Reset on interruption

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        pass

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        audio = self._user_audio_buffer
        self._user_audio_buffer = bytearray()
        if self._needs_turn_complete_message:
            self._needs_turn_complete_message = False
            evt = events.ClientContentMessage.model_validate(
                {"clientContent": {"turnComplete": True}}
            )
            await self.send_client_event(evt)
        if self._transcribe_user_audio and self._context:
            await self._transcribe_audio_queue.put(audio)

    async def _handle_transcribe_user_audio(self, audio, context):
        text = await self._transcribe_audio(audio, context)
        if not text:
            return
        # Sometimes the transcription contains newlines; we want to remove them.
        cleaned_text = text.rstrip("\n")
        logger.debug(f"[Transcription:user] {cleaned_text}")
        await self.push_frame(
            TranscriptionFrame(text=cleaned_text, user_id="user", timestamp=time_now_iso8601()),
            FrameDirection.UPSTREAM,
        )

    async def _transcribe_audio(self, audio, context):
        (text, prompt_tokens, completion_tokens, total_tokens) = await self._transcriber.transcribe(
            audio, context
        )
        if not text:
            return ""
        # The only usage metrics we have right now are for the transcriber LLM. The Live API is free.
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        return text

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OpenAILLMContextFrame):
            logger.info(f"{self} RECEIVED_CONTEXT_FRAME. Context: {frame.context.get_messages_for_logging()}")
            logger.info(f"{self} State on receiving OpenAILLMContextFrame: _api_session_ready={self._api_session_ready}, _run_llm_when_api_session_ready={self._run_llm_when_api_session_ready}, _waiting_for_model_turn_after_tool={self._waiting_for_model_turn_after_tool}, _is_initial_context={self._is_initial_context}")

            # Always update the internal context
            self._context = GeminiMultimodalLiveContext.upgrade(frame.context)
            if frame.context.tools: # Ensure tools are also updated if provided in the context frame
                self._tools = frame.context.tools

            logger.info(f"{self} After upgrading context, last message role: {self._context.messages[-1].get('role') if self._context.messages else 'no messages'}")
            logger.info(f"{self} Outstanding tool IDs: {self._outstanding_tool_ids}")


            if self._waiting_for_model_turn_after_tool:
                logger.info(f"{self} Waiting for model turn after tool submission. Context updated, but no new LLM call initiated by this frame.")
                # If we are waiting for a model turn after _tool_result, it means _tool_result
                # has already sent `turnComplete:True`. We shouldn't send another one from here
                # based on this context update alone. The context is now up-to-date for the *next* interaction.
                return

            # Check if this context contains a *newly added* tool result for an outstanding call
            # This scenario handles the case where the aggregator processes FunctionCallResultFrame
            # and pushes the updated context.
            tool_result_to_process = None
            if self._context.messages and self._outstanding_tool_ids:
                for msg in reversed(self._context.messages): # Check recent messages first
                    if msg.get("role") == "tool" and msg.get("tool_call_id") in self._outstanding_tool_ids:
                        tool_result_to_process = msg
                        break # Process one tool result from context at a time this way

            if tool_result_to_process:
                logger.info(f"{self} Context contains result for outstanding tool {tool_result_to_process.get('tool_call_id')}. Processing via _tool_result.")
                await self._tool_result(tool_result_to_process)
                # After _tool_result, _waiting_for_model_turn_after_tool will be True.
                # The Gemini API has been informed. We should not call _create_response here.
            elif self._is_initial_context:
                logger.info(f"{self} Processing initial context via _create_initial_response.")
                await self._create_initial_response() # This will set self._is_initial_context = False
            elif self._context.messages and self._context.messages[-1].get("role") == "user":
                logger.info(f"{self} Context is a user turn (last message is user). Calling _create_response.")
                # This is a new user turn, or a user utterance after a tool call has been fully processed by the model.
                await self._create_single_response([self._context.messages[-1]]) # Send only the last user message as the new turn
            else:
                # This case might happen if the context ends with an assistant message
                # and no tool calls are pending, or if the context is empty after initial.
                logger.info(f"{self} Context not identified as initial, pending tool response, or new user turn. Last role: {self._context.messages[-1].get('role') if self._context.messages else 'N/A'}. No LLM action taken by this frame.")

        # ... (rest of process_frame conditions as you have them) ...
        elif isinstance(frame, InputAudioRawFrame):
            if not self._audio_input_paused:
                await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        # ... other elif ...
        elif isinstance(frame, LLMMessagesAppendFrame):
            # This is for when the bot needs to say something unprompted by user,
            # or to add a synthetic user/assistant message.
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings) # You might need to re-send config to Gemini if session active
        elif isinstance(frame, LLMSetToolsFrame):
            self._tools = frame.tools # Store new tools
            if self._context: # If context exists, update its tools
                self._context.set_tools(frame.tools)
            await self._update_settings() # Re-send config to Gemini if session active
        else:
            await self.push_frame(frame, direction)

    #
    # websocket communication
    #

    async def send_client_event(self, event):
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        if self._websocket:
            # Here we assume that if we have a websocket, we are connected. We
            # handle disconnections in the send/recv code paths.
            return

        logger.info("Connecting to Gemini service")
        try:
            logger.info(f"Connecting to wss://{self._base_url}")
            uri = f"wss://{self._base_url}?key={self._api_key}"
            self._websocket = await websockets.connect(uri=uri)
            self._receive_task = self.create_task(self._receive_task_handler())
            self._transcribe_audio_task = self.create_task(self._transcribe_audio_handler())

            # Create the basic configuration
            config_data = {
                "setup": {
                    "model": self._model_name,
                    "generation_config": {
                        "frequency_penalty": self._settings["frequency_penalty"],
                        "max_output_tokens": self._settings["max_tokens"],
                        "presence_penalty": self._settings["presence_penalty"],
                        "temperature": self._settings["temperature"],
                        "top_k": self._settings["top_k"],
                        "top_p": self._settings["top_p"],
                        "response_modalities": self._settings["modalities"].value,
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {"voice_name": self._voice_id}
                            },
                            "language_code": self._settings["language"],
                        },
                        "media_resolution": self._settings["media_resolution"].value,
                    },
                    "output_audio_transcription": {},
                }
            }

            # Add context window compression if enabled
            if self._settings.get("context_window_compression", {}).get("enabled", False):
                compression_config = {}
                # Add sliding window (always true if compression is enabled)
                compression_config["sliding_window"] = {}

                # Add trigger_tokens if specified
                trigger_tokens = self._settings.get("context_window_compression", {}).get(
                    "trigger_tokens"
                )
                if trigger_tokens is not None:
                    compression_config["trigger_tokens"] = trigger_tokens

                config_data["setup"]["context_window_compression"] = compression_config

            # Add VAD configuration if provided
            if self._settings.get("vad"):
                vad_config = {}
                vad_params = self._settings["vad"]

                # Only add parameters that are explicitly set
                if vad_params.disabled is not None:
                    vad_config["disabled"] = vad_params.disabled

                if vad_params.start_sensitivity:
                    vad_config["start_of_speech_sensitivity"] = vad_params.start_sensitivity.value

                if vad_params.end_sensitivity:
                    vad_config["end_of_speech_sensitivity"] = vad_params.end_sensitivity.value

                if vad_params.prefix_padding_ms is not None:
                    vad_config["prefix_padding_ms"] = vad_params.prefix_padding_ms

                if vad_params.silence_duration_ms is not None:
                    vad_config["silence_duration_ms"] = vad_params.silence_duration_ms

                # Only add automatic_activity_detection if we have VAD settings
                if vad_config:
                    realtime_config = {"automatic_activity_detection": vad_config}

                    config_data["setup"]["realtime_input_config"] = realtime_config

            config = events.Config.model_validate(config_data)

            # Add system instruction if available
            system_instruction = self._system_instruction or ""
            if self._context and hasattr(self._context, "extract_system_instructions"):
                system_instruction += "\n" + self._context.extract_system_instructions()
            if system_instruction:
                logger.debug(f"Setting system instruction: {system_instruction}")
                config.setup.system_instruction = events.SystemInstruction(
                    parts=[events.ContentPart(text=system_instruction)]
                )

            # Add tools if available
            if self._tools:
                logger.debug(f"Gemini is configuring to use tools{self._tools}")
                config.setup.tools = self.get_llm_adapter().from_standard_tools(self._tools)

            # Send the configuration
            await self.send_client_event(config)

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        logger.info("Disconnecting from Gemini service")
        try:
            self._disconnecting = True
            self._api_session_ready = False
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            if self._transcribe_audio_task:
                await self.cancel_task(self._transcribe_audio_task)
                self._transcribe_audio_task = None
            self._disconnecting = False
            self._outstanding_tool_ids.clear()  # Clear on disconnect
            self._waiting_for_model_turn_after_tool = False
        except Exception as e:
            logger.error(f"{self} error disconnecting: {e}")

    async def _ws_send(self, message):
        # logger.debug(f"Sending message to websocket: {message}")
        try:
            if self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting:
                return
            logger.error(f"Error sending message to websocket: {e}")
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    #
    # inbound server event handling
    # todo: docs link here
    #

    async def _receive_task_handler(self):
        async for message in self._websocket:
            evt = events.parse_server_event(message)
            # logger.debug(f"Received event: {message[:500]}")
            # logger.debug(f"Received event: {evt}")

            if evt.setupComplete:
                await self._handle_evt_setup_complete(evt)
            elif evt.serverContent and evt.serverContent.modelTurn:
                await self._handle_evt_model_turn(evt)
            elif evt.serverContent and evt.serverContent.turnComplete:
                await self._handle_evt_turn_complete(evt)
            elif evt.serverContent and evt.serverContent.outputTranscription:
                await self._handle_evt_output_transcription(evt)
            elif evt.toolCall:
                await self._handle_evt_tool_call(evt)
            elif False:  # !!! todo: error events?
                await self._handle_evt_error(evt)
                # errors are fatal, so exit the receive loop
                return
            else:
                pass

    async def _transcribe_audio_handler(self):
        while True:
            audio = await self._transcribe_audio_queue.get()
            await self._handle_transcribe_user_audio(audio, self._context)

    #
    #
    #

    async def _send_user_audio(self, frame):
        if self._audio_input_paused:
            return
        # Send all audio to Gemini
        evt = events.AudioInputMessage.from_raw_audio(frame.audio, frame.sample_rate)
        await self.send_client_event(evt)
        # Manage a buffer of audio to use for transcription
        audio = frame.audio
        if self._user_is_speaking:
            self._user_audio_buffer.extend(audio)
        else:
            # Keep 1/2 second of audio in the buffer even when not speaking.
            self._user_audio_buffer.extend(audio)
            length = int((frame.sample_rate * frame.num_channels * 2) * 0.5)
            self._user_audio_buffer = self._user_audio_buffer[-length:]

    async def _send_user_video(self, frame):
        if self._video_input_paused:
            return

        now = time.time()
        if now - self._last_sent_time < 1:
            return  # Ignore if less than 1 second has passed

        self._last_sent_time = now  # Update last sent time
        logger.debug(f"Sending video frame to Gemini: {frame}")
        evt = events.VideoInputMessage.from_image_frame(frame)
        await self.send_client_event(evt)

    async def _create_initial_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        messages = self._context.get_messages_for_initializing_history()
        if not messages:
            return

        logger.debug(f"Creating initial response: {messages}")

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": self._inference_on_context_initialization,
                }
            }
        )
        await self.send_client_event(evt)
        if not self._inference_on_context_initialization:
            self._needs_turn_complete_message = True

    async def _create_single_response(self, messages_list):
        # refactor to combine this logic with same logic in GeminiMultimodalLiveContext
        messages = []
        for item in messages_list:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        if not messages:
            return
        logger.debug(f"Creating response: {messages}")

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": True,
                }
            }
        )
        await self.send_client_event(evt)

    async def _tool_result(self, tool_result_message: Dict[str, Any]):
        # tool_result_message is expected to be a dict like:
        # {"role": "tool", "content": "...", "tool_call_id": "...", "name": "optional_tool_name", "function_name": "actual_function_name"}
        tool_call_id = tool_result_message.get("tool_call_id")
        function_name = tool_result_message.get("function_name") or tool_result_message.get("name") # Prefer function_name if available

        if not tool_call_id or not function_name:
            logger.error(f"{self} Tool result message missing 'tool_call_id' or 'function_name'/'name': {tool_result_message}")
            return

        raw_content = tool_result_message.get("content", "")
        try:
            # Content from FunctionCallResultFrame is already a dict or string
            if isinstance(raw_content, dict):
                 parsed_content = raw_content
            elif isinstance(raw_content, str):
                try:
                    parsed_content = json.loads(raw_content)
                except (json.JSONDecodeError, TypeError):
                    # If it's not valid JSON, wrap it as Gemini expects a dict for `response` part
                    parsed_content = {"result_text": raw_content}
            else: # If it's neither dict nor string, wrap it
                parsed_content = {"result_value": str(raw_content)}

        except Exception as e: # Catch any other parsing/conversion error
            logger.error(f"Error processing tool result content '{raw_content}': {e}")
            parsed_content = {"error": f"Failed to process tool result: {e}"}

        # Ensure parsed_content is a dictionary for Gemini's `response` field.
        if not isinstance(parsed_content, dict):
            logger.warning(f"Tool result content '{parsed_content}' is not a dict, wrapping it.")
            parsed_content = {"value": parsed_content}


        response_message_payload = {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": tool_call_id,
                        "name": function_name,
                        "response": parsed_content, # This MUST be a dict for Gemini API
                    }
                ],
            }
        }
        response_message_str = json.dumps(response_message_payload)
        logger.info(f"{self} SENDING TOOL RESPONSE for id={tool_call_id}, name={function_name}: {response_message_str}")
        try:
            if self._websocket:
                await self._websocket.send(response_message_str)
                logger.info(f"{self} Successfully sent tool response to Gemini API for {tool_call_id}")

                if tool_call_id in self._outstanding_tool_ids:
                    self._outstanding_tool_ids.discard(tool_call_id)
                else:
                    logger.warning(f"Tool ID {tool_call_id} was not in _outstanding_tool_ids when result was sent.")


                turn_complete_message = json.dumps({"clientContent": {"turnComplete": True}})
                logger.info(f"{self} SENDING TURN COMPLETE after tool response for {tool_call_id}: {turn_complete_message}")
                await self._websocket.send(turn_complete_message)
                logger.info(f"{self} Successfully sent turnComplete after tool response for {tool_call_id}")
                self._waiting_for_model_turn_after_tool = True # Set flag HERE
            else:
                logger.error(f"{self} WebSocket not available, cannot send tool response for {tool_call_id}.")
        except Exception as e:
            logger.error(f"{self} Error sending tool response or turnComplete for {tool_call_id}: {e}")
            self._outstanding_tool_ids.add(tool_call_id) # Re-add if sending failed, for potential retry
            self._waiting_for_model_turn_after_tool = False # Reset if sending failed

    async def _handle_evt_setup_complete(self, evt):
        previous_state = self._api_session_ready
        self._api_session_ready = True
        logger.info(f"{self} RECEIVED_SETUP_COMPLETE. API session ready. Status changed: {previous_state} -> {self._api_session_ready}")
        if self._run_llm_when_api_session_ready:
            logger.info(f"{self} API session ready and _run_llm_when_api_session_ready is True. Creating initial response.")
            self._run_llm_when_api_session_ready = False
            await self._create_initial_response() # This will set _is_initial_context to False
        else:
            logger.info(f"{self} API session ready but _run_llm_when_api_session_ready is False. Not creating initial response now.")

    async def _handle_evt_model_turn(self, evt):
        self._waiting_for_model_turn_after_tool = False # Model has started its turn
        # ... (rest of your existing _handle_evt_model_turn logic) ...
        logger.info(f"{self} RECEIVED_MODEL_TURN event. Model is generating a response. _waiting_for_model_turn_after_tool reset to False.")
        
        part = evt.serverContent.modelTurn.parts[0]
        if not part:
            logger.warning(f"{self} Model turn contains no parts")
            return

        # part.text is added when `modalities` is set to TEXT; otherwise, it's None
        text = part.text
        if text:
            logger.info(f"{self} Model generated text: {text[:100]}...")
            
            if not self._bot_text_buffer: # First chunk of a new response
                logger.info(f"{self} First text chunk of response, sending LLMFullResponseStartFrame")
                await self.push_frame(LLMFullResponseStartFrame())

            self._bot_text_buffer += text
            await self.push_frame(LLMTextFrame(text=text))

        inline_data = part.inlineData
        if not inline_data:
            return
        if inline_data.mimeType != f"audio/pcm;rate={self._sample_rate}":
            logger.warning(f"Unrecognized server_content format {inline_data.mimeType}")
            return

        audio = base64.b64decode(inline_data.data)
        if not audio:
            return

        if not self._bot_is_speaking:
            self._bot_is_speaking = True
            await self.push_frame(TTSStartedFrame())
            if not self._bot_text_buffer: # If no text was generated, still send start frame
                 await self.push_frame(LLMFullResponseStartFrame())


        self._bot_audio_buffer.extend(audio)
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
        await self.push_frame(frame)

    async def _handle_evt_tool_call(self, evt):
        logger.info(f"{self} RECEIVED_TOOL_CALL event. Model is requesting a tool call.")
        function_calls = evt.toolCall.functionCalls if evt.toolCall else []
        if not function_calls:
            logger.warning(f"{self} Tool call event contains no function calls")
            return
        if not self._context:
            logger.error(f"{self} Function calls are not supported without a context object.")
            return
        logger.info(f"{self} Processing {len(function_calls)} function calls from tool call event")
        for call in function_calls:
            logger.info(f"{self} Calling function: id={call.id}, name={call.name}, args={call.args}")
            self._outstanding_tool_ids.add(call.id)
            try:
                await self.call_function(
                    context=self._context,
                    tool_call_id=call.id,
                    function_name=call.name,
                    arguments=call.args,
                )
                logger.info(f"{self} Successfully initiated call for function {call.name}")
            except Exception as e:
                logger.error(f"{self} Error initiating call for function {call.name}: {e}")
                self._outstanding_tool_ids.discard(call.id)

    async def _handle_evt_turn_complete(self, evt):
        self._waiting_for_model_turn_after_tool = False # Model's turn is complete
        # ... (rest of your existing _handle_evt_turn_complete logic) ...
        logger.info(f"{self} RECEIVED_TURN_COMPLETE event. Model has finished its turn. _waiting_for_model_turn_after_tool reset to False.")
        
        # Mark bot as no longer speaking
        previous_speaking_state = self._bot_is_speaking
        self._bot_is_speaking = False
        logger.info(f"{self} Bot speaking state changed: {previous_speaking_state} -> {self._bot_is_speaking}")
        
        # Log accumulated bot text
        text = self._bot_text_buffer
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"{self} Final bot text buffer on turn complete: {text_preview}")
        self._bot_text_buffer = ""

        # Only push the TTSStoppedFrame if the bot is outputting audio
        # when text is found, modalities is set to TEXT and no audio is produced.
        if self._settings["modalities"] == GeminiMultimodalModalities.AUDIO or not text:
            logger.info(f"{self} Modalities include AUDIO or no text, sending TTSStoppedFrame")
            await self.push_frame(TTSStoppedFrame())
        else:
            logger.info(f"{self} Modalities TEXT only and text present, not sending TTSStoppedFrame.")

        logger.info(f"{self} Sending LLMFullResponseEndFrame to signal end of LLM response")
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_evt_output_transcription(self, evt):
        if not evt.serverContent.outputTranscription:
            return

        # This is the output transcription text when modalities is set to AUDIO.
        # In this case, we push LLMTextFrame and TTSTextFrame to be handled by the
        # downstream assistant context aggregator.
        text = evt.serverContent.outputTranscription.text

        if not text:
            return

        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(TTSTextFrame(text=text))

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GeminiMultimodalLiveContextAggregatorPair:
        """Create an instance of GeminiMultimodalLiveContextAggregatorPair from
        an OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator
                parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User
                aggregator parameters.

        Returns:
            GeminiMultimodalLiveContextAggregatorPair: A pair of context
            aggregators, one for the user and one for the assistant,
            encapsulated in an GeminiMultimodalLiveContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        GeminiMultimodalLiveContext.upgrade(context)
        user = GeminiMultimodalLiveUserContextAggregator(context, params=user_params)

        assistant_params.expect_stripped_words = False
        assistant = GeminiMultimodalLiveAssistantContextAggregator(context, params=assistant_params)
        return GeminiMultimodalLiveContextAggregatorPair(_user=user, _assistant=assistant)
