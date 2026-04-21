"""
Video processing service
"""
import tempfile
import os
import subprocess
import boto3
import time
from typing import List, Dict

class VideoProcessor:
    """Handle video download and audio extraction"""
    
    @staticmethod
    def download_from_s3(s3_bucket: str, s3_key: str, s3_region: str, 
                        aws_access_key: str = None, aws_secret_key: str = None) -> str:
        """Download video from S3"""
        print(f"Downloading video from S3: {s3_bucket}/{s3_key}")
        
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                region_name=s3_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            s3_client = boto3.client('s3', region_name=s3_region)
        
        # Tạo file tạm và đóng ngay để tránh lock trên Windows
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_path = tmp_video.name
        tmp_video.close()  # Đóng file handle ngay
        
        # Download file
        s3_client.download_file(s3_bucket, s3_key, video_path)
        
        print(f"Video downloaded: {video_path}")
        return video_path
    
    @staticmethod
    def extract_audio(video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        print("Extracting audio from video...")
        
        # Tạo đường dẫn audio output
        audio_path = video_path.replace(".mp4", f"_{int(time.time())}.wav")
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Audio codec
            "-ar", "16000",  # Sample rate
            "-ac", "1",  # Mono
            audio_path,
            "-y"  # Overwrite
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        
        print(f"Audio extracted: {audio_path}")
        return audio_path
    
    @staticmethod
    def cleanup_files(*file_paths):
        """Clean up temporary files"""
        for path in file_paths:
            if path and os.path.exists(path):
                os.remove(path)
