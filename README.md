# GLM-ASR Web

A modern, high-performance web interface for Automatic Speech Recognition (ASR), powered by **GLM-ASR** and **Whisper** models. This project provides a user-friendly way to perform real-time transcription and file-based transcription with advanced features like timestamping and multi-model support.

## Features

-   **Real-time Transcription**: Speak directly into your microphone and see the transcription as you talk.
-   **Multi-Model Support**:
    -   **GLM-ASR**: Optimized for mixed Chinese/English conversation (default).
    -   **Whisper Turbo**: Fast and accurate multilingual transcription.
-   **File Transcription**: Upload audio/video files (MP3, WAV, MP4, WebM, M4A, etc.) for batch processing.
-   **Configurable Settings**:
    -   Select Model.
    -   Adjust Timestamp granularity (None, Sentence, Word).
    -   Select Language (for supported models).
-   **Rich UI**:
    -   Visual audio recorder.
    -   Real-time progress bars.
    -   Interactive transcription results with export options (TXT, SRT, JSON).
    -   Dark/Light mode support.

## Prerequisites

-   **Python 3.10+** (Tested with Python 3.13)
-   **Node.js 18+**
-   **FFmpeg** (Required for audio conversion)
    -   Mac: `brew install ffmpeg`

## Installation

The project includes a `Makefile` mechanism for easy setup (via `start.sh` or manual commands).

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd asr-web
    ```

2.  **Install Dependencies**:
    ```bash
    # Install Python (Backend) and Node.js (Frontend) dependencies
    make install
    ```
    *This runs `pip install -r backend/requirements.txt` and `npm install` in `frontend`.*

## Configuration (.env)

The project uses environment variables for configuration. You can customize the host and port in `.env` files.

-   **Backend (`backend/.env`)**:
    ```env
    HOST=0.0.0.0
    PORT=8000
    ```
-   **Frontend (`frontend/.env`)**:
    ```env
    API_HOST=localhost
    API_PORT=8000
    ```

## Usage

1.  **Start the Application**:
    ```bash
    make start
    ```
    This will start both the backend API server and the frontend development server.

2.  **Access the Web Interface**:
    Open your browser and navigate to:
    `http://localhost:5173`

3.  **Start Transcribing**:
    -   **Real-time**: Switch to "Real-time Recording" tab, click "Start Recording", and speak.
    -   **File**: Upload an audio file in the "File Upload" tab and click "Transcribe".

## Project Structure

```
asr-web/
├── backend/                 # Python FastAPI Backend
│   ├── main.py              # Application entry point
│   ├── websocket_handler.py # Real-time transcription logic
│   ├── requirements.txt     # Python dependencies
│   └── ...
├── frontend/                # React + Vite Frontend
│   ├── src/                 # React source code
│   ├── vite.config.js       # Vite configuration
│   └── ...
├── start.sh                 # Startup script
├── Makefile                 # Automation commands
└── README.md                # Project documentation
```

## Technologies

-   **Backend**: Python, FastAPI, WebSockets, Torch, Torchaudio, Transformers
-   **Frontend**: React, Vite, Ant Design
-   **Models**: GLM-ASR, OpenAI Whisper (via HuggingFace/MLX)

## License

[MIT](LICENSE)
