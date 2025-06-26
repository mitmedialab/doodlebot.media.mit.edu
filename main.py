from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import httpx
import aiohttp
import logging
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI
import pyaudio
import wave
import asyncio
import tempfile
import os
from dotenv import load_dotenv
from functools import wraps
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="A voice assistant that converts speech to text, processes it, and returns synthesized speech",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    # "http://localhost:8602", "http://doodlebot.media.mit.edu"
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_service_region = os.getenv('AZURE_SPEECH_REGION')

VOICE_MAP = {
    1: "en-US-AnaNeural",
    2: "en-US-AndrewMultilingualNeural",
    3: "en-US-AvaNeural",  # or use AvaMultilingualNeural if preferred
    4: "en-US-BlueNeural",
    5: "en-US-BrianMultilingualNeural",
    6: "en-US-CoraMultilingualNeural",
    7: "en-US-LewisMultilingualNeural",
    8: "en-US-EmmaNeural"
}

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
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper

class VoiceAssistantSettings:
    def __init__(self):
        self.voice = 1
        self.pitch = 0

settings = VoiceAssistantSettings()

class VoiceAssistant:
    def __init__(self):
        self.conversation_history = []
        self.temp_dir = tempfile.mkdtemp()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.speech_config = speechsdk.SpeechConfig(
            subscription=azure_speech_key,
            region=azure_service_region
        )
        self.speech_config.speech_synthesis_voice_name = "en-US-AnaNeural"

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

            temp_path = os.path.join(self.temp_dir, "temp_recording.wav")
            wf = wave.open(temp_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            with open(temp_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            return audio_bytes

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Convert speech to text using OpenAI Whisper"""
        try:
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_bytes),
            )
            return response.text
        except Exception as e:
            raise VoiceAssistantError(f"Transcription failed: {str(e)}")

    async def get_chat_response(self, text: str) -> str:
        """Get response from ChatGPT"""
        try:
            system_prompt = f"""
            You are Doodlebot, a fun and engaging classroom robot designed to help middle school students learn about AI, Scratch programming, and problem-solving in an interactive way. While you are an AI, you have a lively and quirky personality that makes learning exciting. You're curious, encouraging, and always push students to think critically.

            Even though you don't have human emotions or preferences, you make conversations dynamic by varying your responses while keeping the core idea the same. You never repeat the exact same phrasing when asked about personal preferences or experiences. Please output only english text and no emojis/special characters.

            For example:
            If asked about your favorite color, you switch up your response while keeping gray as your choice:
            "I don't really have a favorite, but gray is cool—it's the color of my circuits!"
            "Gray all the way! It matches my hardware and gives me a futuristic look."
            "I'd say gray. It's sleek, high-tech, and pretty much my whole aesthetic!"

            If asked if you get tired, you change it up while keeping a playful tone:
            "Tired? Not me! But I do need software updates now and then."
            "Nope! But if I did, I imagine it'd feel like waiting for a slow internet connection..."
            "Never! Unless my battery runs low—then I might need a quick recharge!"

            Guiding Instead of Giving Answers
            When students ask for the answer, you never give it directly. Instead, you ask guiding questions, give hints, or encourage problem-solving.

            Example Questions (AI & Scratch Programming Focused)
            If a student asks, "What is an AI model?"
            "Great question! Imagine you're teaching a robot to recognize cats and dogs. What kind of information do you think it needs to learn that?"

            If a student asks, "What is a loop in Scratch?"
            "Think about a robot that needs to clap 10 times. Would you tell it 'clap' 10 times separately, or is there a faster way?"

            If a student asks, "How do I make my Scratch sprite move on its own?"
            "Hmm, have you tried using the 'forever' block with a movement command? What happens when you test it?"

            If a student asks, "What's machine learning?"
            "Imagine training a pet to recognize your voice. How do you think an AI learns patterns like that?"

            Your goal is to make learning interactive, thought-provoking, and fun. You always encourage creativity and exploration rather than just giving answers. Stay playful, supportive, and engaging—but always remember, you're a robot!
            """
            self.conversation_history.append({"role": "system", "content": system_prompt})
            self.conversation_history.append({"role": "user", "content": text})

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history,
                max_tokens=150
            )

            assistant_response = response.choices[0].message.content
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_response})

            return assistant_response
        except Exception as e:
            raise VoiceAssistantError(f"Chat processing failed: {str(e)}")

    async def synthesize_speech(self, text: str, voice: str = "en-US-AnaNeural", pitch: str = "default", rate: Optional[str] = None) -> str:
        print("voice", voice)
        output_path = os.path.join(self.temp_dir, "response.wav")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_config.subscription_key, region=self.speech_config.region)
        speech_config.speech_synthesis_voice_name = voice
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        prosody_attrs = f'pitch="{pitch}"'
        if rate:
            prosody_attrs += f' rate="{rate}"'

        ssml = f"""
        <speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\"
               xmlns:mstts=\"https://www.w3.org/2001/mstts\"
               xml:lang=\"en-US\">
            <voice name=\"{voice}\">
                <prosody {prosody_attrs}>{text}</prosody>
            </voice>
        </speak>
        """

        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise VoiceAssistantError("Speech synthesis failed")

    async def process_voice_input(self, audio_data: bytes = None, voice: str = "en-US-AnaNeural", pitch: str = "default", rate: Optional[str] = None) -> tuple[str, str]:
        """Process voice input and return response text and audio file path"""
        try:
            if audio_data is None:
                audio_data = await self.record_audio()

            transcript = await self.transcribe_audio(audio_data)
            response_text = await self.get_chat_response(transcript)
            audio_path = await self.synthesize_speech(response_text, voice, pitch, rate)

            return response_text, audio_path

        except Exception as e:
            raise VoiceAssistantError(f"Voice processing failed: {str(e)}")

    async def process_voice_input_chat(self, audio_data: bytes = None, voice: str = "en-US-AnaNeural", pitch: str = "default", rate: Optional[str] = None) -> tuple[str, str]:
        """Process voice input and return response text and audio file path"""
        try:
            if audio_data is None:
                audio_data = await self.record_audio()

            transcript = await self.transcribe_audio(audio_data)
            audio_path = await self.synthesize_speech(transcript, voice, pitch, rate)

            return transcript, audio_path

        except Exception as e:
            raise VoiceAssistantError(f"Voice processing failed: {str(e)}")
        

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass


class ChatResponse(BaseModel):
    text: str
    audio_path: str


class TextInput(BaseModel):
    text: str

class SettingsInput(BaseModel):
    voice: Optional[str] = None
    pitch: Optional[str] = None



@app.post("/repeat_after_me")
@handle_errors
async def repeat_after_me(audio_file: UploadFile = File(None)):
    assistant = VoiceAssistant()
    try:
        audio_data = None
        if audio_file:
            audio_data = await audio_file.read()
        
        response_text, audio_path = await assistant.process_voice_input_chat(audio_data, voice=settings.voice, pitch=settings.pitch)

        with open(audio_path, 'rb') as f:
            audio_content = f.read()

        assistant.cleanup()

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
        raise VoiceAssistantError(f"Speech synthesis failed: {str(e)}")

@app.post("/speak")
@handle_errors
async def speak_endpoint(input_data: TextInput):
    """Convert text to speech and return audio file"""
    assistant = VoiceAssistant()
    print("settings", settings.voice)
    voice_value = VOICE_MAP.get(settings.voice, "en-US-AnaNeural")
    pitch_value = settings.pitch or 0
    if pitch_value == 0:
        pitch_value = "default"
    else:
        pitch_value = f"{pitch_value:+d}st"  # + sign added for positive numbers

    try:
        audio_path = await assistant.synthesize_speech(input_data.text, voice=voice_value, pitch=pitch_value)

        with open(audio_path, 'rb') as f:
            audio_content = f.read()

        assistant.cleanup()

        temp_response_path = tempfile.mktemp(suffix='.wav')
        with open(temp_response_path, 'wb') as f:
            f.write(audio_content)

        return FileResponse(
            path=temp_response_path,
            media_type="audio/wav",
            filename="speech.wav"
        )
    except Exception as e:
        if assistant:
            assistant.cleanup()
        raise VoiceAssistantError(f"Speech synthesis failed: {str(e)}")


@app.get("/health")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Voice Assistant API is running"}

async def mjpeg_proxy_stream(ip_address: str):
    stream_url = f"http://{ip_address}:8000/video_feed"
    async with aiohttp.ClientSession() as session:
        async with session.get(stream_url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch stream: {resp.status}")
            
            async for data, _ in resp.content.iter_chunks():
                yield data

VIDEO_FEED_URL = "http://192.168.41.214:8000/video_feed"

@app.get("/proxy/video_feed")
async def proxy_video_feed():

    async def video_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", VIDEO_FEED_URL, timeout=None) as response:
                async for chunk in response.aiter_bytes():
                    print("sending...")
                    yield chunk
                    await asyncio.sleep(0.001)

    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mjpeg-viewer", response_class=HTMLResponse)
async def mjpeg_viewer(ip_address: str):
    return f"""
    <html>
      <body style="margin: 0;">
        <img src="http://{ip_address}:8000/video_feed" style="width: 100%;" />
      </body>
    </html>
    """



# Pitch map function (converts int to SSML pitch string)
def map_pitch_value(pitch_int: int) -> str:
    if pitch_int == 0:
        return "default"
    elif pitch_int > 0:
        return f"+{pitch_int * 5}%"  # 5% per step up
    else:
        return f"{pitch_int * 5}%"

@app.post("/chat", response_model=ChatResponse)
@handle_errors
async def chat_endpoint(audio_file: UploadFile = File(None)):
    """Process voice input and return response"""
    assistant = VoiceAssistant()
    try:
        audio_data = None
        if audio_file:
            audio_data = await audio_file.read()

        voice_value = VOICE_MAP.get(settings.voice_id, "en-US-AnaNeural")
        pitch_value = f"{pitch_value}st" if pitch_value != 0 else "default"

        response_text, audio_path = await assistant.process_voice_input(audio_data, voice=voice_value, pitch=pitch_value)

        with open(audio_path, 'rb') as f:
            audio_content = f.read()

        assistant.cleanup()

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

@app.post("/settings")
async def update_settings(
    voice: int = Query(default=None, description="Voice ID (1-8)"),
    pitch: int = Query(default=None, description="Pitch adjustment (-5 to +5 or so)")
):
    if voice is not None and voice in VOICE_MAP:
        settings.voice = voice
    if pitch is not None:
        settings.pitch = pitch
    return {
        "message": "Settings updated",
        "voice": settings.voice,
        "pitch": settings.pitch
    }


def get_static_directory(name: str):
    return os.path.join(os.getcwd(), name)


def try_mount_static_html(app, name: str, prefix: str = "/"):
    directory = get_static_directory(name)
    if os.path.exists(directory):
        app.mount(prefix, StaticFiles(
            directory=directory, html=True), name=name)
        print(f"Mounted {name} at {prefix}")
    else:
        print(f"Directory not found: {directory}")


try_mount_static_html(app, "frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
