"""Video metadata, frame extraction, classification, aggregation, and temporal tracking."""

from ai.video.aggregator import (
    AGGREGATION_METHOD,
    VideoAggregationResult,
    aggregate_frame_predictions,
)
from ai.video.temporal import (
    TEMPORAL_METHOD,
    analyze_temporal_consistency,
    associate_detections,
    bbox_iou,
    spatial_match_score,
)
from ai.video.frame_analyzer import analyze_extracted_frames
from ai.video.frame_extractor import ExtractedFrame, VideoExtractionResult, extract_frames
from ai.video.metadata import VideoMetadata, read_video_metadata
from ai.video.video_config import VideoConfig, get_video_config

__all__ = [
    "AGGREGATION_METHOD",
    "ExtractedFrame",
    "VideoAggregationResult",
    "VideoConfig",
    "VideoExtractionResult",
    "VideoMetadata",
    "aggregate_frame_predictions",
    "TEMPORAL_METHOD",
    "analyze_temporal_consistency",
    "associate_detections",
    "bbox_iou",
    "spatial_match_score",
    "analyze_extracted_frames",
    "extract_frames",
    "get_video_config",
    "read_video_metadata",
]
