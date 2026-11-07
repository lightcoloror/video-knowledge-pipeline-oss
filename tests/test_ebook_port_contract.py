from video_knowledge_pipeline.config import service_url


def test_default_ebook_http_contract_uses_current_9241_port() -> None:
    assert (
        service_url("ebook_markdown_pipeline_http")
        == "http://127.0.0.1:9241/call"
    )
