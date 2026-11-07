from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .path_defaults import video_tool_lab_root


@dataclass(frozen=True)
class ToolCandidate:
    name: str
    category: str
    url: str
    install_hint: str
    best_for: str
    reuse_role: str
    visual_sampling: int
    transcript_timestamps: int
    obsidian_fit: int
    local_first: int
    windows_fit: int
    research_fit: int
    maturity: int
    friction: int
    notes: str
    command_names: tuple[str, ...]

    @property
    def learning_score(self) -> int:
        return (
            self.visual_sampling * 3
            + self.transcript_timestamps * 2
            + self.obsidian_fit * 3
            + self.local_first * 2
            + self.windows_fit
            + self.research_fit * 3
            + self.maturity
            - self.friction * 2
        )


CANDIDATES: tuple[ToolCandidate, ...] = (
    ToolCandidate(
        "vidwise",
        "ready-to-use extraction artifact",
        "https://pypi.org/project/vidwise/",
        'python -m pip install "vidwise[fast]"',
        "Local/YouTube video to transcript, SRT, key frames, and Markdown visual guide.",
        "Primary first trial for video learning notes; import guide.md and transcript into Obsidian/video-knowledge.",
        4,
        4,
        5,
        4,
        4,
        4,
        3,
        2,
        "Pixel-difference key frame selection; Markdown output with relative images; optional AI guide.",
        ("vidwise",),
    ),
    ToolCandidate(
        "vidclaude",
        "Claude Code oriented evidence assembly",
        "https://pypi.org/project/vidclaude/",
        "python -m pip install vidclaude",
        "Adaptive/shot-aware frames, faster-whisper transcript, optional OCR, evidence.md.",
        "Best research-grade local extraction layer after applying CPU env/patch; strongest for learning videos with slides/UI/text.",
        5,
        5,
        4,
        4,
        3,
        5,
        2,
        3,
        "Tested locally: scene changes, transcript, OCR, timeline, evidence.md. Needs CPU env/patch on this machine.",
        ("vidclaude",),
    ),
    ToolCandidate(
        "peepshow",
        "fast local frame/OCR timeline",
        "https://www.peepshow.dev/",
        "npx peepshow or node <npm-cache>/peepshow/dist/cli.js on Windows",
        "Scene/fps/transnet frame extraction, OCR, optional transcript, HTML report, and many sinks.",
        "Best quick local materializer: generate frame/OCR/report artifacts, then import them into video-knowledge or Obsidian.",
        5,
        2,
        5,
        4,
        3,
        4,
        3,
        2,
        "Tested locally via npm package: Windows .cmd wrapper fails because it expects /bin/sh, but direct node dist/cli.js works; OCR works with PEEPSHOW_TESSERACT.",
        ("peepshow",),
    ),
    ToolCandidate(
        "vidlizer",
        "structured JSON scene flow",
        "https://pypi.org/project/vidlizer/",
        "python -m pip install vidlizer",
        "Scene-by-scene JSON flow with what happened, visible text, people, and changes.",
        "Use when a local/cloud VLM provider is already configured and structured JSON is more important than human-readable study notes.",
        4,
        3,
        3,
        4,
        2,
        4,
        3,
        4,
        "Installed locally; doctor reports no local Ollama/LM Studio/OpenRouter and missing mlx-whisper, so it is not ready on this machine.",
        ("vidlizer",),
    ),
    ToolCandidate(
        "Video Vision MCP",
        "MCP perception layer",
        "https://videovisionmcp.com/",
        "npx -y @oamaestro/video-vision-mcp",
        "Agent workflow over URLs/local videos with frames, scenes, captions, and local Whisper.",
        "Good MCP experiment after CLI tools; maturity needs hands-on verification.",
        4,
        4,
        2,
        4,
        3,
        3,
        1,
        3,
        "Advertises scene changes and local Whisper; repo maturity appears uncertain.",
        ("vvmp", "video-vision-mcp"),
    ),
    ToolCandidate(
        "MCPTube",
        "YouTube knowledge base",
        "https://0xchamin.github.io/mcptube/",
        "pipx install mcptube --python python3.12",
        "YouTube videos as AI-queryable knowledge base with transcript, frames, reports.",
        "Try for YouTube learning playlists, not as the generic local video backbone.",
        4,
        4,
        3,
        3,
        3,
        3,
        2,
        3,
        "Good if source videos are mostly YouTube; less universal than vidwise/vidclaude.",
        ("mcptube",),
    ),
    ToolCandidate(
        "Video-RAG",
        "architecture reference",
        "https://video-rag.github.io/",
        "use as architecture/reference before full repo integration",
        "OCR + ASR + object detection + frame retrieval architecture for long-video QA.",
        "Reference design for future adapter; not first choice for personal daily learning.",
        5,
        4,
        1,
        3,
        1,
        5,
        2,
        5,
        "Best conceptual architecture; likely too heavy as a first personal tool.",
        ("video-rag",),
    ),
    ToolCandidate(
        "FunClip",
        "transcript-driven clipping",
        "https://github.com/modelscope/FunClip",
        "git clone https://github.com/modelscope/FunClip.git",
        "Chinese/English ASR, timestamped subtitles, speaker or text segment clipping.",
        "Use as ASR/timestamp signal source for Chinese speech-heavy videos.",
        1,
        5,
        2,
        5,
        3,
        2,
        4,
        4,
        "Strong transcript-time alignment; weak visual evidence.",
        ("funclip",),
    ),
    ToolCandidate(
        "WhisperX",
        "ASR timestamp component",
        "https://github.com/m-bain/whisperX",
        "python -m pip install whisperx",
        "Word-level timestamps and diarization for transcript semantic timepoint selection.",
        "Component only; pair with visual sampler and evidence-card layer.",
        0,
        5,
        1,
        4,
        3,
        3,
        4,
        3,
        "Useful when transcript timing quality matters; does not see video.",
        ("whisperx",),
    ),
)


def tool_matrix() -> list[dict]:
    rows = []
    for candidate in sorted(CANDIDATES, key=lambda item: item.learning_score, reverse=True):
        row = asdict(candidate)
        row["learning_score"] = candidate.learning_score
        row["installed_paths"] = installed_paths(candidate)
        row["installed"] = bool(row["installed_paths"])
        rows.append(row)
    return rows


def installed_paths(candidate: ToolCandidate) -> list[str]:
    paths = [path for name in candidate.command_names if (path := shutil.which(name))]
    lab_scripts = video_tool_lab_root() / ".venv" / "Scripts"
    for name in candidate.command_names:
        for suffix in (".exe", ".cmd", ".bat", ""):
            candidate_path = lab_scripts / f"{name}{suffix}"
            if candidate_path.exists():
                paths.append(str(candidate_path))
    if candidate.name == "peepshow":
        npx_root = Path.home() / "AppData/Local/npm-cache/_npx"
        if npx_root.exists():
            for dist_cli in npx_root.glob("*/node_modules/peepshow/dist/cli.js"):
                paths.append(str(dist_cli))
    return sorted(set(paths))


def recommended_trial_order() -> list[dict]:
    rows = tool_matrix()
    for index, row in enumerate(rows, start=1):
        row["trial_order"] = index
    return rows

