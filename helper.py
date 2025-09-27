import asyncio
import json
import logging
import os
import time
from typing import Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from openai import AsyncOpenAI
import numpy as np
from fastapi import WebSocketDisconnect, WebSocket
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Tracks the state of an active conversation"""
    user_id: str
    conversation_history: list
    current_llm_task: Optional[asyncio.Task] = None
    current_tts_task: Optional[asyncio.Task] = None
    interrupt_event: asyncio.Event = None
    is_processing: bool = False
    last_activity: float = 0
    
    def __post_init__(self):
        if self.interrupt_event is None:
            self.interrupt_event = asyncio.Event()
        self.last_activity = time.time()

class SpeechPipelineServer:
    """
    Production-grade WebSocket server for speech-to-speech pipeline
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Active connections and conversation states
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        
        # Configuration
        self.config = {
            "openai_model": "gpt-4o-mini",
            "tts_voice": "alloy",
            "tts_speed": 1.0,
            "max_conversation_history": 20,
            "stream_chunk_size": 1024,
            "whisper_model_size": "base",
            "session_timeout": 60,  # 5 minutes
        }
        
        # Start cleanup task
        # asyncio.create_task(self.cleanup_inactive_sessions())
    
    
    async def handle_websocket(self, websocket: WebSocket, user_id: str):
        """Handle WebSocket connection for a user"""
        try:
            await websocket.accept()
            
            # Initialize conversation state
            self.active_connections[user_id] = websocket
            self.conversation_states[user_id] = ConversationState(
                user_id=user_id,
                conversation_history=[
                    {"role": "system", "content": "You are a helpful AI assistant engaged in a natural conversation. Keep responses conversational, concise, and engaging. Respond as if you're speaking aloud."}
                ]
            )
            
            logger.info(f"User {user_id} connected")
            
            # Send connection confirmation
            await self.send_message(websocket, {
                "type": "connection_status",
                "status": "connected",
                "user_id": user_id
            })
            
            # Listen for messages
            async for message in websocket.iter_text():
                await self.process_message(user_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"User {user_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
        finally:
            await self.cleanup_user_session(user_id)
    
    async def process_message(self, user_id: str, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            state = self.conversation_states.get(user_id)
            
            if not state:
                logger.error(f"No conversation state for user {user_id}")
                return
            
            state.last_activity = time.time()
            message_type = data.get("type")
            
            if message_type == "speech":
                await self.handle_speech_input(user_id, data)
            elif message_type == "audio":
                await self.handle_audio_input(user_id, data)
            elif message_type == "interrupt":
                await self.handle_interrupt(user_id)
            elif message_type == "config":
                await self.handle_config_update(user_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from user {user_id}: {message}")
        except Exception as e:
            logger.error(f"Error processing message from user {user_id}: {e}")
    
    async def handle_speech_input(self, user_id: str, data: dict):
        """Handle speech input from Web Speech API"""
        text = data.get("text", "").strip()
        voice = data.get("voice", self.config["tts_voice"])
        
        if not text:
            return
        
        state = self.conversation_states[user_id]
        websocket = self.active_connections[user_id]
        
        logger.info(f"Speech input from {user_id}: {text}")
        
        # Cancel any ongoing processing
        await self.handle_interrupt(user_id)
        
        # Add user message to conversation history
        state.conversation_history.append({"role": "user", "content": text})
        
        # Trim conversation history if too long
        if len(state.conversation_history) > self.config["max_conversation_history"]:
            # Keep system message and trim from the beginning
            system_msg = state.conversation_history[0]
            state.conversation_history = [system_msg] + state.conversation_history[-(self.config["max_conversation_history"]-1):]
        
        # Start AI processing
        state.is_processing = True
        state.current_llm_task = asyncio.create_task(
            self.generate_ai_response(user_id, voice)
        )
    
    async def handle_audio_input(self, user_id: str, data: dict):
        """Handle raw audio input for Whisper transcription"""
        if not self.whisper_model:
            await self.send_error(user_id, "Whisper model not available")
            return
        
        try:
            # Decode base64 audio data
            import base64
            audio_data = base64.b64decode(data.get("audio", ""))
            
            # Convert to numpy array (assuming 16kHz PCM)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_np, language="en")
            text = result["text"].strip()
            
            if text:
                # Process as speech input
                await self.handle_speech_input(user_id, {
                    "text": text,
                    "voice": data.get("voice", self.config["tts_voice"])
                })
            
        except Exception as e:
            logger.error(f"Whisper transcription error for user {user_id}: {e}")
            await self.send_error(user_id, "Transcription failed")
    
    async def handle_interrupt(self, user_id: str):
        """Handle interrupt signal - cancel ongoing AI processing"""
        state = self.conversation_states.get(user_id)
        if not state:
            return
        
        logger.info(f"Interrupt signal from user {user_id}")
        
        # Set interrupt event
        state.interrupt_event.set()
        
        # Cancel ongoing tasks
        if state.current_llm_task and not state.current_llm_task.done():
            state.current_llm_task.cancel()
            
        if state.current_tts_task and not state.current_tts_task.done():
            state.current_tts_task.cancel()
        
        # Reset state
        state.is_processing = False
        
        # Clear interrupt event for next use
        await asyncio.sleep(0.1)  # Small delay to ensure cancellation
        state.interrupt_event.clear()
        
        # Notify client
        await self.send_message(self.active_connections[user_id], {
            "type": "interrupt_acknowledged"
        })
    
    async def handle_config_update(self, user_id: str, data: dict):
        """Handle configuration updates from client"""
        config_updates = data.get("config", {})
        
        # Update allowed configuration
        allowed_updates = ["tts_voice", "tts_speed"]
        for key, value in config_updates.items():
            if key in allowed_updates:
                self.config[key] = value
        
        logger.info(f"Config updated for user {user_id}: {config_updates}")
    
    # In your SpeechPipelineServer class in Python

    # In your SpeechPipelineServer class in Python
# This is the final version that correctly handles interruptions.

    async def generate_ai_response(self, user_id: str, voice: str):
        """
        Generate AI response and stream TTS sequentially. This simple version is the most stable.
        """
        state = self.conversation_states[user_id]
        
        try:
            await self.send_message(self.active_connections[user_id], {"type": "ai_status", "status": "processing"})
            
            full_response_text = ""
            llm_response_stream = self.stream_llm_response(state.conversation_history)

            async for chunk in llm_response_stream:
                if state.interrupt_event.is_set():
                    logger.info("Interrupt detected during LLM stream. Halting.")
                    return
                
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    full_response_text += content_chunk
                    
                    while self.is_sentence_complete(full_response_text):
                        sentence = self.extract_complete_sentence(full_response_text)
                        full_response_text = full_response_text[len(sentence):].strip()

                        if any(char.isalnum() for char in sentence):
                            # Await the streaming function directly. It will now handle its own interruption.
                            await self.stream_tts_audio(user_id, sentence, voice)
                            # Check for interrupt again immediately after the sentence is spoken
                            if state.interrupt_event.is_set():
                                logger.info("Interrupt detected after sentence stream. Halting.")
                                return
                        else:
                            logger.warning(f"Skipping invalid sentence for TTS: '{sentence}'")

            final_text = full_response_text.strip()
            if final_text and not state.interrupt_event.is_set():
                if any(char.isalnum() for char in final_text):
                    await self.stream_tts_audio(user_id, final_text, voice)

        except asyncio.CancelledError:
            logger.info(f"AI response generation cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"AI response generation error for user {user_id}: {e}")
            await self.send_error(user_id, "AI response generation failed")
        finally:
            if self.active_connections.get(user_id):
                await self.send_message(self.active_connections[user_id], {"type": "ai_status", "status": "completed"})
            state.is_processing = False
    
    async def stream_llm_response(self, conversation_history: list) -> AsyncGenerator:
        """Stream LLM response from OpenAI"""
        try:
            stream = await self.openai_client.chat.completions.create(
                model=self.config["openai_model"],
                messages=conversation_history,
                stream=True,
                max_tokens=500,
                temperature=0.7
            )
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def stream_tts_audio(self, user_id: str, text: str, voice: str):
        """
        Generate and stream TTS audio, now fully interruptible even during the OpenAI API call.
        """
        state = self.conversation_states[user_id]
        websocket = self.active_connections.get(user_id)
        if not websocket: return

        # =======================================================================
        # --- THE INTERRUPTION FIX ---
        # =======================================================================
        
        # Create a task for the potentially long-running API call
        api_call_task = asyncio.create_task(self.openai_client.audio.speech.create(
            model="tts-1", voice=voice, input=text, response_format="mp3", speed=self.config["tts_speed"]
        ))
        
        # Create a listener task that waits for the interrupt event
        interrupt_listener = asyncio.create_task(state.interrupt_event.wait())
        
        try:
            # Wait for EITHER the API call to finish OR the interrupt to happen
            done, pending = await asyncio.wait(
                {api_call_task, interrupt_listener},
                return_when=asyncio.FIRST_COMPLETED
            )

            # If the interrupt listener finished, it means we were interrupted.
            if interrupt_listener in done:
                logger.info(f"TTS API call interrupted by user for text: '{text}'")
                # Cancel the API call task that was still running in the background
                api_call_task.cancel()
                return # <-- Exit the function immediately.
            
            # If we get here, the API call finished successfully. Get the result.
            response = api_call_task.result()
            
        except Exception as e:
            logger.error(f"Error during interruptible API call for TTS: {e}")
            return
        finally:
            # Always clean up the listener task
            interrupt_listener.cancel()
        # =======================================================================

        # The rest of the function proceeds only if not interrupted.
        # This part already has interrupt checks in its loop.
        if len(response.content) < 100:
            logger.error(f"Received invalid audio from OpenAI. Size: {len(response.content)} bytes.")
            return

        await self.send_message(websocket, {"type": "tts_status", "status": "generating", "text": text})

        complete_base64 = base64.b64encode(response.content).decode('utf-8')
        chunk_size = self.config["stream_chunk_size"]
        
        for i in range(0, len(complete_base64), chunk_size):
            if state.interrupt_event.is_set():
                logger.info("TTS streaming loop interrupted.")
                return
            
            await self.send_message(websocket, {"type": "audio_chunk", "data": complete_base64[i:i + chunk_size]})
            await asyncio.sleep(0.01)

        if not state.interrupt_event.is_set():
            await self.send_message(websocket, {"type": "tts_stream_end"})
                
    def is_sentence_complete(self, text: str) -> bool:
        """Check if text contains a complete sentence"""
        sentence_endings = ['.', '!', '?', ';']
        return any(ending in text for ending in sentence_endings)
    
    def extract_complete_sentence(self, text: str) -> str:
        """Extract the first complete sentence from text"""
        sentence_endings = ['.', '!', '?', ';']
        
        for i, char in enumerate(text):
            if char in sentence_endings:
                # Include the punctuation and return
                return text[:i + 1].strip()
        
        return ""
    
    async def send_message(self, websocket: WebSocket, message: dict):
        """
        Takes a dictionary, converts it to a JSON string, and sends it via WebSocket text.
        This function no longer handles any special encoding.
        """
        try:
            # No more if/else. Every message is treated the same way.
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def send_error(self, user_id: str, error_message: str):
        """Send error message to client"""
        websocket = self.active_connections.get(user_id)
        if websocket:
            await self.send_message(websocket, {
                "type": "error",
                "message": error_message
            })
    
    async def cleanup_user_session(self, user_id: str):
        """Clean up user session on disconnect"""
        try:
            # Cancel ongoing tasks
            if user_id in self.conversation_states:
                state = self.conversation_states[user_id]
                
                if state.current_llm_task and not state.current_llm_task.done():
                    state.current_llm_task.cancel()
                    
                if state.current_tts_task and not state.current_tts_task.done():
                    state.current_tts_task.cancel()
            
            # Remove from active connections and states
            self.active_connections.pop(user_id, None)
            self.conversation_states.pop(user_id, None)
            
            logger.info(f"Cleaned up session for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session for user {user_id}: {e}")
    
    async def cleanup_inactive_sessions(self):
        """Periodically clean up inactive sessions"""
        while True:
            try:
                current_time = time.time()
                timeout = self.config["session_timeout"]
                
                inactive_users = []
                for user_id, state in self.conversation_states.items():
                    if current_time - state.last_activity > timeout:
                        inactive_users.append(user_id)
                
                for user_id in inactive_users:
                    logger.info(f"Cleaning up inactive session: {user_id}")
                    await self.cleanup_user_session(user_id)
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)