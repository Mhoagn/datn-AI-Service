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

# ==========================================
# Async Job Schemas
# ==========================================

class JobStartResponse(BaseModel):
    job_id: str
    status: str  # "processing"

class JobResult(BaseModel):
    status: str
    transcript_segments: List[TranscriptSegment]
    full_text: str
    summary: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # "processing" | "completed" | "failed"
    result: Optional[JobResult] = None
    error: Optional[str] = None
