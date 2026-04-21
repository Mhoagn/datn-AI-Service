"""
AI Service - Speech to Text & Text Summarization
Optimized with model caching and modular structure
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import whisperx
import tempfile
import os

from models import model_manager
from services import VideoProcessor, TranscriptService, SummaryService
from schemas import ProcessVideoRequest, ProcessVideoResponse, TranscriptSegment
from config import HOST, PORT, WHISPER_MODEL, QWEN_MODEL, DEVICE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - models will be loaded on-demand"""
    print("=" * 60)
    print("Starting AI Service...")
    print(f"WhisperX Model: {WHISPER_MODEL}")
    print(f"Qwen Model: {QWEN_MODEL}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Don't load models at startup to save memory
    # They will be loaded on-demand
    
    print("=" * 60)
    print("AI Service ready!")
    print("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    model_manager.cleanup()

app = FastAPI(
    title="AI Service - Speech to Text & Summarization",
    description="WhisperX + Qwen 2.5 for meeting transcription and summarization",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# WEB UI FOR TESTING
# ==========================================
@app.get("/", response_class=HTMLResponse)
def get_webpage():
    """Web interface for testing audio transcription"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Service - Test WhisperX</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; text-align: center; margin-top: 50px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            button { padding: 15px 30px; font-size: 18px; cursor: pointer; border: none; border-radius: 5px; margin: 10px; }
            .record-btn { background: #4CAF50; color: white; }
            .stop-btn { background: #f44336; color: white; }
            .status { margin: 20px 0; font-size: 16px; }
            .result { background: #f4f4f4; padding: 20px; border-radius: 5px; text-align: left; margin-top: 20px; }
            pre { white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎙️ Test WhisperX Transcription</h2>
            <button id="recordBtn" class="record-btn">Bắt đầu Ghi âm</button>
            <p class="status" id="status">Trạng thái: Sẵn sàng</p>
            <div class="result">
                <strong>Kết quả:</strong>
                <pre id="result">Chưa có kết quả</pre>
            </div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            const recordBtn = document.getElementById('recordBtn');
            const status = document.getElementById('status');
            const resultBox = document.getElementById('result');

            recordBtn.onclick = async () => {
                if (!isRecording) {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    isRecording = true;
                    recordBtn.innerText = "🛑 Dừng & Phân tích";
                    recordBtn.className = "stop-btn";
                    status.innerText = "Trạng thái: Đang thu âm...";
                    audioChunks = [];

                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    
                    mediaRecorder.onstop = async () => {
                        status.innerText = "Trạng thái: Đang xử lý với WhisperX...";
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const formData = new FormData();
                        formData.append("file", audioBlob, "recording.webm");

                        try {
                            const response = await fetch('/transcribe', { method: 'POST', body: formData });
                            const data = await response.json();
                            resultBox.innerText = JSON.stringify(data, null, 2);
                            status.innerText = "Trạng thái: Hoàn tất!";
                        } catch (error) {
                            resultBox.innerText = "Lỗi: " + error.message;
                            status.innerText = "Trạng thái: Lỗi!";
                        }
                    };
                } else {
                    mediaRecorder.stop();
                    isRecording = false;
                    recordBtn.innerText = "Bắt đầu Ghi âm";
                    recordBtn.className = "record-btn";
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==========================================
# API: TRANSCRIBE AUDIO FILE
# ==========================================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        audio = whisperx.load_audio(tmp_path)
        whisper_model = model_manager.get_whisper_model()
        result = whisper_model.transcribe(audio, batch_size=1)
        
        return {
            "status": "success",
            "filename": file.filename,
            "text": result["segments"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================================
# API: PROCESS VIDEO FROM S3
# ==========================================
@app.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(request: ProcessVideoRequest):
    """
    Download video from S3, extract audio, transcribe, and summarize
    """
    video_path = None
    audio_path = None
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting video processing...")
        print(f"S3: {request.s3_bucket}/{request.s3_key}")
        print(f"{'='*60}\n")
        
        # 1. Download video from S3
        print("Step 1/4: Downloading video from S3...")
        video_path = VideoProcessor.download_from_s3(
            request.s3_bucket,
            request.s3_key,
            request.s3_region,
            request.aws_access_key,
            request.aws_secret_key
        )
        
        # 2. Extract audio
        print("\nStep 2/4: Extracting audio from video...")
        audio_path = VideoProcessor.extract_audio(video_path)
        
        # 3. Transcribe
        print("\nStep 3/4: Transcribing audio with WhisperX...")
        transcript_result = TranscriptService.transcribe(audio_path)
        
        # Calculate duration from segments
        duration_seconds = 0
        if transcript_result["segments"]:
            last_segment = transcript_result["segments"][-1]
            duration_seconds = last_segment.get("end", 0)
        
        print(f"Video duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
        
        # Unload WhisperX to free VRAM for Qwen
        print("\nFreeing GPU memory...")
        model_manager.unload_whisper_model()
        
        # 4. Summarize with duration info
        print("\nStep 4/4: Summarizing text with Qwen...")
        summary = SummaryService.summarize(transcript_result["full_text"], duration_seconds)
        
        # 5. Build response
        segments = [
            TranscriptSegment(**seg) 
            for seg in transcript_result["segments"]
        ]
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Segments: {len(segments)}")
        print(f"Text length: {len(transcript_result['full_text'])} chars")
        print(f"{'='*60}\n")
        
        return ProcessVideoResponse(
            status="success",
            transcript_segments=segments,
            full_text=transcript_result["full_text"],
            summary=summary
        )
        
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"ERROR in process_video:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        print("Cleaning up temporary files...")
        VideoProcessor.cleanup_files(video_path, audio_path)

# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_manager._models_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
