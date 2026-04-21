"""
Transcript service using WhisperX
"""
import whisperx
from typing import List, Dict
from models import model_manager

class TranscriptService:
    """Handle audio transcription"""
    
    @staticmethod
    def transcribe(audio_path: str, batch_size: int = 8) -> Dict:
        """
        Transcribe audio using WhisperX on CPU
        CPU mode: Slower but stable (no GPU memory issues)
        """
        print("🎤 Transcribing audio with WhisperX on CPU...")
        print("   (This may take a while...)")
        
        try:
            audio = whisperx.load_audio(audio_path)
            whisper_model = model_manager.get_whisper_model()
            
            # Simple transcribe - WhisperX will handle everything
            # No need to disable VAD, CPU has enough memory
            result = whisper_model.transcribe(
                audio, 
                batch_size=batch_size,
                language="vi"  # Specify Vietnamese
            )
            
            segments = []
            full_text_parts = []
            
            # Check if we have segments
            if "segments" not in result or len(result["segments"]) == 0:
                print("⚠️ No speech detected in audio.")
                return {
                    "segments": [],
                    "full_text": "[Khong phat hien duoc giong noi trong audio]"
                }
            
            for seg in result["segments"]:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })
                full_text_parts.append(seg.get("text", "").strip())
            
            full_text = " ".join(full_text_parts)
            
            if not full_text.strip():
                full_text = "[Khong co noi dung van ban]"
            
            print(f"✅ Transcription complete!")
            print(f"   Segments: {len(segments)}")
            print(f"   Text length: {len(full_text)} chars")
            print(f"   Sample: {full_text[:100]}...")
            
            return {
                "segments": segments,
                "full_text": full_text
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "segments": [],
                "full_text": f"[Loi khi transcript: {str(e)}]"
            }
