"""Source-checkout compatibility entry; canonical implementation is packaged."""
from video_knowledge_pipeline.context_manifest_consumer_cli import (
    _shared, consume_handoff, decide_asr, main,
)

if __name__ == "__main__":
    raise SystemExit(main())
