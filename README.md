# Voice Assistant Backend

## Overview
This FastAPI backend handles speech-to-text, natural language processing, and text-to-speech conversion using OpenAI's Whisper, GPT models, and Azure's Speech Services.

## Setup & Prerequisites

### File Structure
```
.
├── main.py           # Main application file
├── test.py          # Test file generator
├── .env             # Environment variables
└── README.md        # This file
```

### Environment Variables
Create a `.env` file in the root directory with the following:
```
OPENAI_API_KEY=your_openai_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
```

#### VoiceAssistant Class
- Manages recording, transcription, chat, and playback
- Handles conversation state
- Coordinates API interactions

#### FastAPI Integration
- Single endpoint: `/chat`
- Supports both streaming and non-streaming responses
- Swagger UI available at `/docs`

### Main Functions
1. `record_audio()`: Captures microphone input
2. `process_audio()`: Converts speech to text using Whisper
3. `get_chat_response()`: Generates response using ChatGPT
4. `synthesize_speech()`: Converts text to speech
5. `play_audio()`: Plays the response

## Implementation Flow
1. User initiates chat through endpoint
2. System records audio
3. Audio processed through Whisper
4. Response generated via ChatGPT
5. Response converted to speech
6. Audio played back to user

## Testing
- Run test file generator: `python3 test.py`
- Navigate to `/docs` to test the `/chat` endpoint
- Basic error handling tests included
- Core functionality tests for each component

## Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your `.env` file with the required API keys

3. Start the server:
```bash
uvicorn main:app --reload
```

4. Navigate to `http://localhost:8000/docs` to test the `/chat` endpoint

5. Generate test WAV file:
```bash
python3 test.py
```