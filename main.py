from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
import openai
from openai import OpenAI
import pyaudio
import wave
import tempfile
import os
from dotenv import load_dotenv
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="A voice assistant that converts speech to text, processes it, and returns synthesized speech",
    version="1.0.0"
)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_service_region = os.getenv('AZURE_SPEECH_REGION')

class VoiceAssistantError(Exception):
    """Custom exception for Voice Assistant errors"""
    pass

def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except VoiceAssistantError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper

class VoiceAssistant:
    def __init__(self):
        self.conversation_history = []
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Azure Speech config
        self.speech_config = speechsdk.SpeechConfig(
            subscription=azure_speech_key,
            region=azure_service_region
        )
        
        # Audio recording config
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 5

    async def record_audio(self) -> bytes:
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            # Save to temporary WAV file
            temp_path = os.path.join(self.temp_dir, "temp_recording.wav")
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Read the file as bytes
            with open(temp_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            return audio_bytes
            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Convert speech to text using OpenAI Whisper"""
        print("🔊 Transcribing audio...")
        try:
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_bytes),
            )
            print("✅ Transcription complete!")
            print(response.text + "\n")
            return response.text
        except Exception as e:
            raise VoiceAssistantError(f"Transcription failed: {str(e)}")

    async def get_chat_response(self, text: str) -> str:
        """Get response from ChatGPT"""
        print("💬 Getting chat response...")
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history,
                max_tokens=150
            )
            
            assistant_response = response.choices[0].message.content
            print("✅ Chat response complete!")
            print(assistant_response + "\n")
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        except Exception as e:
            raise VoiceAssistantError(f"Chat processing failed: {str(e)}")

    async def synthesize_speech(self, text: str) -> str:
        """Convert text to speech using Azure"""
        try:
            output_path = os.path.join(self.temp_dir, "response.wav")
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("✅ Speech synthesis complete!")
                return output_path
            else:
                raise VoiceAssistantError("Speech synthesis failed")
                
        except Exception as e:
            raise VoiceAssistantError(f"Speech synthesis failed: {str(e)}")

    async def process_voice_input(self, audio_data: bytes = None) -> tuple[str, str]:
        """Process voice input and return response text and audio file path"""
        try:
            # Record audio if not provided
            if audio_data is None:
                audio_data = await self.record_audio()
            
            # Convert speech to text
            transcript = await self.transcribe_audio(audio_data)
            
            # Get ChatGPT response
            response_text = await self.get_chat_response(transcript)
            
            # Convert response to speech
            audio_path = await self.synthesize_speech(response_text)
            
            return response_text, audio_path
            
        except Exception as e:
            raise VoiceAssistantError(f"Voice processing failed: {str(e)}")

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

# API Models
class ChatResponse(BaseModel):
    text: str
    audio_path: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Voice Assistant API is running"}

@app.post("/chat", response_model=ChatResponse)
@handle_errors
async def chat_endpoint(audio_file: UploadFile = File(None)):
    """Process voice input and return response"""
    assistant = VoiceAssistant()
    try:
        audio_data = None
        if audio_file:
            # Read uploaded file
            audio_data = await audio_file.read()
        
        response_text, audio_path = await assistant.process_voice_input(audio_data)
        
        # Read the audio file into memory before cleanup
        with open(audio_path, 'rb') as f:
            audio_content = f.read()
            
        # Clean up files
        assistant.cleanup()
        
        # Create a new temporary file for the response
        temp_response_path = tempfile.mktemp(suffix='.wav')
        with open(temp_response_path, 'wb') as f:
            f.write(audio_content)
        
        return FileResponse(
            path=temp_response_path,
            media_type="audio/wav",
            headers={"text-response": response_text},
            filename="response.wav"
        )
    except Exception as e:
        if assistant:
            assistant.cleanup()
        raise VoiceAssistantError(f"Chat processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)