"""Pydantic v2 domain models for the recap-video pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_LICENSES = (
    "public_domain",
    "cc0",
    "cc_by",
    "cc_by_sa",
    "owned",
    "licensed",
)
REJECTED_LICENSE_PREFIXES = (
    "cc_by_nc",
    "cc_by_nd",
    "all_rights_reserved",
    "unknown",
)
MOODS = (
    "hook",
    "funny",
    "tense",
    "emotional",
    "action",
    "cliffhanger",
    "reveal",
    "calm",
)


class Beat(BaseModel):
    """A single narrative beat in the recap script."""

    beat_id: int = Field(ge=1)
    narration: str = Field(min_length=1)
    source_window_start_sec: float = Field(ge=0)
    source_window_end_sec: float = Field(gt=0)
    visual_cue: str = Field(min_length=1)
    mood: Literal[
        "hook",
        "funny",
        "tense",
        "emotional",
        "action",
        "cliffhanger",
        "reveal",
        "calm",
    ]
    kind: Literal["story", "commentary"]
    audio_duration_sec: Optional[float] = None
    audio_frames: Optional[int] = None

    @model_validator(mode="after")
    def end_after_start(self) -> Beat:
        """Ensure the source window end is strictly after the start."""
        if self.source_window_end_sec <= self.source_window_start_sec:
            raise ValueError(
                f"Beat {self.beat_id}: end ({self.source_window_end_sec}) "
                f"must be > start ({self.source_window_start_sec})"
            )
        return self


class Shot(BaseModel):
    """A single video shot mapped to a beat."""

    beat_id: int = Field(ge=1)
    shot_index: int = Field(ge=0)
    src_start_sec: float = Field(ge=0)
    src_end_sec: float = Field(gt=0)
    speed_factor: float = Field(ge=0.5, le=2.0)
    frames: int = Field(ge=1)
    match_score: Optional[float] = None
    flag: Optional[Literal["low_match", "motion_only", "manual"]] = None


class ProjectManifest(BaseModel):
    """Top-level manifest describing a recap video project."""

    schema_version: int = 1
    project_slug: str = Field(pattern=r"^[a-z0-9_]{3,40}$")
    style: Literal["commentary", "recap"]
    part_number: int = Field(ge=0, le=10)
    title: str = Field(max_length=75)
    banner_text: str
    description: str
    hashtags: list[str] = Field(min_length=1)
    source_title: str
    attribution_block: str = ""
    beats: list[Beat] = Field(min_length=1)
    shots: list[Shot] = Field(default_factory=list)

    @field_validator("banner_text")
    @classmethod
    def banner_word_count(cls, v: str) -> str:
        """Banner text must be 3-6 words."""
        words = v.split()
        if not (3 <= len(words) <= 6):
            raise ValueError(
                f"banner_text must be 3-6 words, got {len(words)}"
            )
        return v


class RightsRecord(BaseModel):
    """Copyright / license record for the source material."""

    work_title: str = Field(min_length=1)
    creator: str = Field(min_length=1)
    year: int
    license: Literal[
        "public_domain", "cc0", "cc_by", "cc_by_sa", "owned", "licensed"
    ]
    license_url: Optional[str] = None
    source_page_url: str = Field(min_length=1)
    commercial_use_allowed: bool
    derivatives_allowed: bool
    attribution_text: Optional[str] = None
    pd_basis: Optional[str] = None
    pd_verified_by_human: bool = False
    license_document_url: Optional[str] = None
    owner_statement: Optional[str] = None
    soundtrack_rights_note: Optional[str] = None
    exclude_head_sec: Optional[float] = None
    exclude_tail_sec: Optional[float] = None
    mute_source_audio: bool = False
    verified_by: str = Field(min_length=1)
    verified_on: str = Field(min_length=1)


class ApprovalRecord(BaseModel):
    """Human approval gate for a project."""

    approved: bool = False
    approved_by: Optional[str] = None
    approved_on: Optional[str] = None


class QueueItem(BaseModel):
    """A single item in the processing queue."""

    slug: str = Field(pattern=r"^[a-z0-9_]{3,40}$")
    video_source_url: str
    srt_source_url: Optional[str] = None
    style: Literal["commentary", "recap"] = "commentary"
    split_part: int = Field(ge=0, le=10, default=0)
    max_duration: int = 175
    publish_target: str = "all_private"
    auto_approve: bool = False
    status: Literal["pending", "done", "failed", "deferred", "awaiting_pd_verification"] = "pending"
    error: Optional[str] = None
    completed_on: Optional[str] = None


class QueueFile(BaseModel):
    """Container for the full processing queue."""

    schema_version: int = 1
    items: list[QueueItem] = Field(default_factory=list)


class RunReport(BaseModel):
    """Post-run report written by the orchestrator."""

    slug: str
    status: Literal["success", "partial", "failed"]
    exit_code: int = 0
    stages_completed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_sec: Optional[float] = None
    source_sha256: Optional[str] = None
    output_files: list[str] = Field(default_factory=list)
    publish_results: dict = Field(default_factory=dict)
    deinterlace_applied: bool = False
    bgm_mood: Optional[str] = None


class PlatformStatus(BaseModel):
    """Publish status for a single platform."""

    published: bool = False
    url: Optional[str] = None
    published_at: Optional[str] = None
    privacy: Optional[str] = None


class HistoryRecord(BaseModel):
    """One entry in the published-history ledger."""

    slug: str
    work_title: str
    normalized_title: str
    source_id: str = ""
    source_url: str = ""
    source_sha256: str = ""
    status: Literal["rendered", "published_partial", "published_all", "queued", "awaiting_pd_verification"] = "rendered"
    platforms: dict[str, PlatformStatus] = Field(
        default_factory=lambda: {
            "youtube": PlatformStatus(),
            "instagram": PlatformStatus(),
        }
    )
    created_at: str = ""
    updated_at: str = ""


class PublishedHistory(BaseModel):
    """Root model for published_history.json."""

    schema_version: int = 1
    records: list[HistoryRecord] = Field(default_factory=list)


class CharacterEntry(BaseModel):
    name: str
    short_role: str


class SeriesPartPlan(BaseModel):
    part_number: int = Field(..., ge=1)
    total_parts: int = Field(..., ge=1)
    window_start_sec: float = Field(..., ge=0.0)
    window_end_sec: float = Field(..., gt=0.0)
    arc_goal: str
    previous_cliffhanger: str = ""
    ending_hook: str
    forbidden_future_spoiler: str = ""


class SeriesBlueprint(BaseModel):
    base_slug: str
    work_title: str
    total_parts: int
    characters: list[CharacterEntry] = Field(default_factory=list)
    parts: list[SeriesPartPlan]


class CalendarSlot(BaseModel):
    day_number: int = Field(..., ge=1, le=365)
    slot_id: str
    base_slug: str
    work_title: str
    year: Optional[int] = None
    part_number: int = Field(..., ge=1)
    total_parts: int = Field(..., ge=1)
    source_provider: str
    runner_target: Literal["github_actions", "personal_mac"] = "github_actions"
    license: Literal["cc0", "cc_by", "cc_by_sa", "public_domain"]
    license_url: str
    source_page_url: str
    video_url: str
    srt_url: Optional[str] = None
    attribution_text: str
    duration_sec: float
    likes: int = 0
    views: int = 0
    editorial_score: float = Field(..., ge=0.0, le=100.0)
    viral_hook_angle: str
    exclude_head_sec: Optional[float] = None
    pd_verified_by_human: bool = False
    status: Literal["scheduled", "rendered", "published", "skipped", "awaiting_pd_verification"] = "scheduled"


class ReleaseCalendar365(BaseModel):
    schema_version: int = 1
    generated_at: str
    total_slots: int
    slots: list[CalendarSlot]


HostEmotion = Literal[
    "neutral", "laughing", "shocked", "horrified",
    "smug", "hyped", "sad", "facepalm"
]
MovieGenre = Literal["comedy", "horror_comedy", "action_scifi", "emotional_drama"]


class ChapterSlice(BaseModel):
    part_number: int = Field(..., ge=1)
    total_parts: int = Field(..., ge=1)
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., gt=0.0)
    duration_sec: float = Field(..., gt=0.0)


class CommentaryCue(BaseModel):
    cue_id: int = Field(..., ge=1)
    start_sec: float = Field(..., ge=0.0)
    max_duration_sec: float = Field(..., gt=0.5)
    text: str
    emotion: HostEmotion
    pause_emotion: HostEmotion = "neutral"


class ChapterCommentaryScript(BaseModel):
    genre: MovieGenre
    voice_name: str
    voice_rate: str
    chapter_title: str
    cliffhanger_summary: str = ""
    cues: list[CommentaryCue]
