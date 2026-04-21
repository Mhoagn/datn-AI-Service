"""
Pydantic schemas for request/response
"""
from pydantic import BaseModel
from typing import Optional, List

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

class ProcessVideoRequest(BaseModel):
    s3_bucket: str
    s3_key: str
    s3_region: str
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

class ProcessVideoResponse(BaseModel):
    status: str
    transcript_segments: List[TranscriptSegment]
    full_text: str
    summary: str
