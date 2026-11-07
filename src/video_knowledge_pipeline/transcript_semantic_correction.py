
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .numeric_normalization import number_evidence_map
from .storage import bundle_write_lock, read_json, write_json
from .model_task_gateway import model_task_api_call
from .model_api_settings import resolve_model_api_provider_config
from .model_business_authorization import (
    create_business_child_consent,
    validate_model_business_authorization,
)
from .trusted_model_connector import execute_consented_model_task
from .text_llm_gateway import extract_json_document, resolve_text_provider_config
from .transcript import format_timestamp, parse_transcript
from .transcript_speakers import (
    cue_speaker,
    speaker_display_name,
    speaker_label_map,
    speaker_payload,
)
from .smart_summary_input_pack import _mapping_text as _visual_mapping_text
from .visual_evidence import (
    TEMPORAL_CONTENT_FIELDS,
    VISUAL_CONTENT_FIELDS,
    temporal_evidence_state,
    visual_evidence_state,
    understanding_source_selector,
    visual_text_source_selector,
    visual_recognition_provenance,
)
from .transcript_semantic_transaction import commit_transcript_semantic_artifacts
from .transcript_semantic_state import (
    DISCOVERY_SCAN_RECEIPT_NAME,
    DISCOVERY_SCAN_RECEIPT_SCHEMA,
    semantic_candidate_resolution,
    semantic_discovery_scan_state,
)

PACK_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_pack.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_result.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_validation.v1"
CLOSURE_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_closure.v1"
DECISION_LEDGER_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_decision_ledger.v1"
IMPACT_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_impact.v1"
STATUS_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_status.v1"
CORRECTED_SCHEMA = "video_knowledge_pipeline.source_arbitrated_transcript.v1"
CANDIDATE_DISCOVERY_SCHEMA = "video_knowledge_pipeline.transcript_semantic_candidate_discovery.v1"
CANDIDATE_SUGGESTIONS_SCHEMA = "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1"
STRICT_MODEL_CONTRACT_MARKER = "VKP_STRICT_TRANSCRIPT_SEMANTIC_CORRECTION_V1"
STRICT_DECISION_REQUIRED_KEYS = {
    "candidate_id": "string",
    "action": "string",
    "correction_type": "string",
    "original_text": "string",
    "corrected_text": "string",
    "confidence": "number",
    "rationale": "string",
    "evidence_ids": "array",
    "human_confirmed": "boolean",
    "needs_human_review": "boolean",
}
STRICT_DECISION_NONEMPTY_KEYS = [
    "candidate_id",
    "action",
    "correction_type",
    "original_text",
    "rationale",
    "evidence_ids",
]

SEMANTIC_ANCHOR_TYPES = {
    "negation",
    "condition",
    "exception",
    "person_name",
    "organization",
    "date_time",
    "money",
    "percentage",
}
VALID_TYPES = {
    "term",
    "proper_noun",
    "number",
    "action",
    "concept",
    "ordinary_word",
    "ordinary_homophone",
    "collocation_anomaly",
    "punctuation",
    "segment_boundary",
    *SEMANTIC_ANCHOR_TYPES,
}
HIGH_RISK_TYPES = {"number", "punctuation", "segment_boundary", *SEMANTIC_ANCHOR_TYPES}
ACTION_HINT_RE = re.compile(r"点击|打开|选择|输入|复制|粘贴|注册|登录|导入|导出|下载|上传|配置|安装|运行|执行|复盘|成交|跟进|筛选|确认|提交|保存")
ACTION_VERB_RE = re.compile(r"点击|打开|选择|输入|复制|粘贴|注册|登录|导入|导出|下载|上传|配置|安装|运行|执行|复盘|成交|跟进|筛选|确认|提交|保存")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|小时|分钟|k|K|w|W|万|亿|元|刀|%|％|年|月|天|秒)?")
CHINESE_FACT_VALUE_RE = re.compile(r"[零一二三四五六七八九十百千万亿两]+\s*(?:万|亿|元|刀|%|％|年|月|日|天|小时|分钟|秒|步|个|次|条|位|家|块)")
GENERIC_CHINESE_QUANTIFIER_RE = re.compile(
    r"^(?:一|两)(?:个|次|天|条|位|家|块|步|份|项|种|段|遍|轮|组|张|句|层|类)$"
)
NEGATION_ANCHOR_RE = re.compile(r"不得|不能|不可|不会|无法|未曾|并非|禁止|拒绝|没有|没|无|未|非|不")
CONDITION_ANCHOR_RE = re.compile(r"只要|只有|除非|如果|假如|若|当[^，。；;]{0,16}时|在[^，。；;]{0,16}情况下|前提")
EXCEPTION_ANCHOR_RE = re.compile(r"责任除外|除外|例外|除非|不包括|不含|包含|仅限|但是|但")
PERSON_NAME_ANCHOR_RE = re.compile(r"[一-鿿]{2,4}(?:老师|先生|女士|经理|博士|教授|主任)")
ORGANIZATION_ANCHOR_RE = re.compile(
    r"[一-鿿A-Za-z0-9·]{2,32}(?:人寿|保险|公司|集团|银行|大学|学院|医院|中心|机构|平台)"
)
DATE_TIME_ANCHOR_RE = re.compile(
    r"(?:\d{2,4}年(?:\d{1,2}月)?(?:\d{1,2}日)?|\d{1,2}月\d{1,2}日|\d{1,2}[:：]\d{2}(?::\d{2})?|(?:今天|明天|昨天|本周|下周|上周|本月|下月|上月))"
)
MONEY_ANCHOR_RE = re.compile(
    r"(?:[\¥￥$]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:万元|亿元|元|块钱|块|美元|港元|人民币|刀))"
)
PERCENTAGE_ANCHOR_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|百分之)")
ODD_SPACING_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\s+){2,}[A-Za-z](?![A-Za-z])")
COMPOUND_SPACED_ASCII_RE = re.compile(r"\b(?:[A-Za-z]\s+){1,}[A-Za-z]\s+[A-Za-z][A-Za-z0-9\-]{2,}\b")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
ASCII_PHRASE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-]{2,}){1,3}\b")
SUPPORT_STOP_TOKENS = {"title", "video", "source", "description", "summary", "platform", "page", "metadata", "http", "https", "课件", "画面", "字幕", "显示", "讲师"}
FILLER_RE = re.compile(r"^(所以啊|然后呢|就是说|这个这个|那个那个|嗯+|啊+|呃+|对吧|是不是)[，,。\s]*$")
MOJIBAKE_RE = re.compile(r"[\ufffd]{2,}|(?:锟斤拷){1,}|(?:Ã|Â|æ|ç|é|å|�)")
BOUNDARY_MARKER_RE = re.compile(r"第一|第二|第三|第四|第五|首先|其次|然后|所以|但是|如果|因为|最后|总结|接下来|另外|比如|比如说")
PUNCTUATION_RE = re.compile(r"[，。！？；：、,.!?;:]")
DEICTIC_OR_SCREEN_REF_RE = re.compile(r"这里|这个|那个|这块|这部分|看一下|看这里|屏幕|画面|上面|下面|左边|右边|重点|很重要|注意")
GENERIC_SUPPORT_PHRASES = {"画面显示", "讲师演示", "工具名", "步骤", "概念", "重点", "结论", "流程", "画面出现"}
SOURCE_RELIABILITY_WEIGHTS = {
    "human_note": 90,
    "structured_visual": 75,
    "ocr": 70,
    "visual_understanding": 65,
    "temporal_visual": 65,
    "platform_subtitle": 50,
    "embedded_subtitle": 50,
    "explicit_domain_lexicon": 60,
    "secondary_asr": 45,
    "asr_or_subtitle": 40,
    "tagger": 30,
    "page_metadata": 20,
    "unknown": 5,
}
SOURCE_RELIABILITY_NOTES = {
    "human_note": "人工标注/复核，默认最高可信，但仍保留审计。",
    "structured_visual": "图文结构化结果，适合证明屏幕文字、表格、公式、代码。",
    "ocr": "OCR/ebook 屏幕文字证据，适合证明画面中出现的实体和数字。",
    "visual_understanding": "多模态单帧理解，适合证明画面对象和界面状态。",
    "temporal_visual": "连续片段理解，适合证明动作、步骤和状态变化。",
    "platform_subtitle": "平台字幕可能也是 ASR，作为中等证据，不能单独压过 OCR/视觉/人工证据。",
    "embedded_subtitle": "视频内嵌/外挂字幕作为中等证据，仍需与 ASR/OCR/视觉互证。",
    "explicit_domain_lexicon": "版本化显式领域词库；必须再有至少一份独立 ASR 精确支持，不能单独覆盖主稿。",
    "secondary_asr": "独立第二 ASR 假设；与主 ASR 的差异只生成冲突候选，不能单独覆盖主稿。",
    "asr_or_subtitle": "当前主 transcript/ASR 来源，默认支持原文。",
    "tagger": "打标器提供时间轴/重点线索，通常作为低到中等权重辅助证据。",
    "page_metadata": "标题、简介、网页信息，适合提示主题和专名，但通常不能单独改写局部 transcript。",
}
LLM_CONFLICT_REASONS = {
    "visual_text_differs_from_transcript",
    "subtitle_text_differs_from_transcript",
    "platform_subtitle_differs_from_transcript",
    "embedded_subtitle_differs_from_transcript",
    "tagger_text_differs_from_transcript",
    "ordinary_word_conflict_between_asr_and_subtitle",
    "ordinary_word_conflict_between_dual_asr",
    "ordinary_word_conflict_between_asr_and_visual_text",
    "ordinary_word_conflict_between_asr_and_tagger",
    "deictic_or_low_information_transcript_with_support_concept",
}
LOW_EVIDENCE_HEURISTIC_REASONS = {
    "contains_number_or_amount",
    "action_or_step_word_in_transcript",
    "punctuation_or_segment_boundary_uncertain",
    "fragmented_or_semantically_weak_phrase",
}
LLM_SUPPORT_SOURCE_TYPES = {
    "human_note",
    "structured_visual",
    "ocr",
    "visual_understanding",
    "temporal_visual",
    "platform_subtitle",
    "embedded_subtitle",
    "secondary_asr",
    "tagger",
}

TIMELINE_ALIGNMENT_WINDOW_SECONDS = 0.25
DISCOVERY_MAX_SEGMENTS = 40
SEMANTIC_CONTEXT_MAX_SECONDS = 30.0
SEMANTIC_CONTEXT_MAX_CUES_PER_SIDE = 2
DIRECT_SEMANTIC_EVIDENCE_TYPES = {
    "human_note",
    "ocr",
    "structured_visual",
    "visual_understanding",
    "temporal_visual",
    "platform_subtitle",
    "embedded_subtitle",
    "secondary_asr",
}

SIDE_SOURCE_MANIFEST_KEYS = [
    ("asr_secondary_transcripts", "secondary_asr"),
    ("asr_secondary_transcript", "secondary_asr"),
    ("qwen3_asr_transcript_json", "secondary_asr"),
    ("fun_asr_nano_transcript_json", "secondary_asr"),
    ("platform_subtitle", "platform_subtitle"),
    ("platform_subtitle_path", "platform_subtitle"),
    ("source_subtitle", "platform_subtitle"),
    ("source_subtitle_path", "platform_subtitle"),
    ("subtitle_path", "platform_subtitle"),
    ("bilibili_subtitle", "platform_subtitle"),
    ("subtitle_json", "platform_subtitle"),
    ("subtitle_srt", "platform_subtitle"),
    ("embedded_subtitle_vtt", "embedded_subtitle"),
    ("embedded_subtitle_srt", "embedded_subtitle"),
    ("embedded_subtitle_json", "embedded_subtitle"),
    ("embedded_subtitle_path", "embedded_subtitle"),
    ("embedded_subtitle", "embedded_subtitle"),
]
SIDE_SOURCE_ROOT_FILES = [
    ("platform-subtitle.json", "platform_subtitle"),
    ("platform-subtitle.srt", "platform_subtitle"),
    ("platform-subtitle.vtt", "platform_subtitle"),
    ("subtitle.json", "platform_subtitle"),
    ("subtitle.srt", "platform_subtitle"),
    ("subtitle.vtt", "platform_subtitle"),
    ("source-subtitle.json", "platform_subtitle"),
    ("source-subtitle.srt", "platform_subtitle"),
    ("source-subtitle.vtt", "platform_subtitle"),
    ("embedded-subtitle.json", "embedded_subtitle"),
    ("embedded-subtitle.srt", "embedded_subtitle"),
    ("embedded-subtitle.vtt", "embedded_subtitle"),
]
METADATA_TEXT_KEYS = (
    "title",
    "video_title",
    "source_title",
    "page_title",
    "description",
    "source_description",
    "page_description",
    "summary",
    "source_url",
    "platform",
)
TAGGER_KEYS = ("tags", "tagger_labels", "tagger_tags", "tagger_annotations", "tagger_visual_summary", "tagger_time_axis")

DRAFT_CANONICAL_CORRECTIONS = {
    "m c p": ("MCP", 0.97, "常见技术缩写，字母间隔 ASR 结果可安全规范化。"),
    "n p c": ("NPC", 0.94, "常见英文缩写，字母间隔 ASR 结果可安全规范化。"),
    "a i": ("AI", 0.96, "常见技术缩写，字母间隔 ASR 结果可安全规范化。"),
    "a p p": ("app", 0.93, "中文语境中常见英文词被逐字母转写，可规范化为 app。"),
    "a a i": ("AI", 0.90, "中文语境中常见技术缩写被重复/间隔转写，可规范化为 AI。"),
    "tiktok": ("TikTok", 0.96, "跨境/短视频语境下的平台名，大小写可安全规范化。"),
    "titok": ("TikTok", 0.93, "跨境/短视频语境下的 TikTok 常见 ASR 漏字。"),
    "shopify": ("Shopify", 0.95, "跨境电商语境下的平台名，大小写可安全规范化。"),
    "whatsapp": ("WhatsApp", 0.95, "跨境电商语境下的通信工具名，大小写可安全规范化。"),
    "bgm": ("BGM", 0.94, "短视频语境下的常见缩写，大小写可安全规范化。"),
    "toc": ("ToC", 0.90, "商业语境下的 ToC 缩写，大小写可安全规范化。"),
    "tob": ("ToB", 0.90, "商业语境下的 ToB 缩写，大小写可安全规范化。"),
    "playright": ("Playwright", 0.93, "浏览器自动化语境下的高频工具名，ASR 常误作 playright。"),
    "playright client": ("Playwright client", 0.93, "浏览器自动化语境下的 Playwright client 工具名。"),
    "chrom": ("Chrome", 0.91, "浏览器自动化语境下指 Chrome，ASR 截断为 chrom。"),
    "stay hand": ("Stagehand", 0.91, "浏览器自动化工具 Stagehand 常被 ASR 拆成 stay hand。"),
    "u i task": ("UI-TARS", 0.92, "桌面/浏览器操作模型 UI-TARS 常被 ASR 误听成 u i task。"),
    "u i tars": ("UI-TARS", 0.96, "UI-TARS 字母间隔 ASR 结果可安全规范化。"),
    "page agent": ("PageAgent", 0.91, "浏览器自动化语境下的 PageAgent 工具名。"),
    "javascript": ("JavaScript", 0.95, "技术名词大小写规范化。"),
}

SAFE_ACRONYM_NORMALIZATIONS = {"AI", "SEO", "SKU", "APP", "app"}

DRAFT_REVIEW_SUGGESTIONS = {
    "open client": "可能是 OpenClaw，但仅凭 ASR 和标题证据不足，建议结合画面/网页/人工复核。",
    "token": "可能是技术术语 token，也可能无需纠错；保留到语义复核。",
    "bug": "可能是普通英文词 bug，不自动纠正。",
}

DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS = {
    "买虫": ("买重", "保险课程里讨论保单查缺补漏时，常见表达是是否买重，ASR 可能误成买虫。"),
    "买虫的": ("买重的", "保险课程里讨论保单查缺补漏时，常见表达是是否有买重的保障。"),
    "二则一": ("二择一", "销售话术/封闭问题语境下常见术语是二择一。"),
    "二则一的方式": ("二择一的方式", "销售话术/封闭问题语境下常见术语是二择一。"),
    "同意心": ("同理心", "销售沟通/客户理解语境下常见表达是同理心。"),
    "明晚八点o": ("明晚八点 OK", "预约确认语境下客户常说 OK，SenseVoice 可能把 OK 识别成字母 o。"),
    "明晚八点O": ("明晚八点 OK", "预约确认语境下客户常说 OK，SenseVoice 可能把 OK 识别成字母 O。"),
    "明晚八点 o": ("明晚八点 OK", "预约确认语境下客户常说 OK，SenseVoice 可能把 OK 识别成独立字母 o。"),
    "明晚八点 O": ("明晚八点 OK", "预约确认语境下客户常说 OK，SenseVoice 可能把 OK 识别成独立字母 O。"),
    "保证方案": ("保障方案", "保险方案语境下应为保障方案；至少还需独立 ASR、画面或显式词库证据。"),
    "第一批兑": ("第一梯队", "产品市场定位语境下常见表达为第一梯队。"),
    "受险": ("寿险", "保险责任语境下常见险种为寿险。"),
    "申雇的责任": ("身故的责任", "寿险责任语境下常见表达为身故的责任。"),
    "身购的责任": ("身故的责任", "寿险责任语境下常见表达为身故的责任。"),
    "报应公司": ("保险公司", "公司类型语境下常见表达为保险公司。"),
    "同讯会议": ("腾讯会议", "线上方案讲解工具名应为腾讯会议。"),
    "重极线": ("重疾险", "保险产品语境下常见险种为重疾险。"),
    "助你显": ("重疾险", "保险产品语境下重疾险的常见 ASR 同音误识别。"),
    "中级显得": ("重疾险的", "保险保额语境下重疾险的常见 ASR 同音误识别。"),
    "手上沟通": ("首次沟通", "销售流程语境下常见阶段为首次沟通。"),
    "出自沟通": ("初次沟通", "销售流程语境下常见阶段为初次沟通。"),
    "守着沟通": ("首次沟通", "销售流程语境下常见阶段为首次沟通。"),
    "指导说": ("知道说", "该上下文表达为要知道说，属于常见同音误识别。"),
    "不注意说保费": ("不至于说保费", "预算沟通语境下表达为不至于说保费出来。"),
    "最大的限度的保证": ("最大的限度的保障", "保险配置语境下应为保障。"),
    "\u5408\u4fdd\u4eba\u5458": ("\u6838\u4fdd\u4eba\u5458", "\u4fdd\u9669\u6838\u4fdd\u6d41\u7a0b\u8bed\u5883\u4e0b\u5e38\u89c1\u89d2\u8272\u4e3a\u6838\u4fdd\u4eba\u5458\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u97f3\u9891\u3001OCR \u6216\u4eba\u5de5\u786e\u8ba4\u3002"),
    "\u805a\u5b9d": ("\u62d2\u4fdd", "\u6838\u4fdd\u7ed3\u8bba\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u62d2\u4fdd\u7684\u540c\u97f3\u8bef\u8bc6\u522b\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u5305\u4f53": ("\u6807\u4f53", "\u4fdd\u9669\u627f\u4fdd\u5206\u7c7b\u8bed\u5883\u4e0b\u5e38\u89c1\u672f\u8bed\u4e3a\u6807\u4f53\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u505c\u7626": ("\u505c\u552e", "\u4fdd\u9669\u4ea7\u54c1\u72b6\u6001\u8bed\u5883\u4e0b\u5e38\u89c1\u8868\u8fbe\u4e3a\u505c\u552e\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u8ddf\u79d1\u6280\u8bb2": ("\u8ddf\u5ba2\u6237\u8bb2", "\u65b9\u6848\u8bb2\u89e3\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u8ddf\u5ba2\u6237\u8bb2\u7684\u8fde\u7eed\u8bef\u8bc6\u522b\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u4f4f\u4f60\u68c0": ("\u91cd\u75be\u9669", "\u4fdd\u9669\u9669\u79cd\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u91cd\u75be\u9669\u7684\u540c\u97f3\u8bef\u8bc6\u522b\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u84dd\u5c3e\u708e": ("\u9611\u5c3e\u708e", "\u5065\u5eb7\u544a\u77e5\u6216\u7406\u8d54\u6848\u4f8b\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u9611\u5c3e\u708e\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u805a\u8d54": ("\u62d2\u8d54", "\u4fdd\u9669\u7406\u8d54\u7ed3\u8bba\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u62d2\u8d54\u7684\u540c\u97f3\u8bef\u8bc6\u522b\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u5174\u8d77\u70b9": ("\u5174\u8da3\u70b9", "\u5ba2\u6237\u6c9f\u901a\u8bed\u5883\u4e0b\u5e38\u89c1\u8868\u8fbe\u4e3a\u5174\u8da3\u70b9\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
    "\u7537\u6027\u7684\u5b9d\u5b9d": ("\u7537\u6027\u7684\u5b9d\u7238", "\u5ba2\u6237\u753b\u50cf\u8bed\u5883\u4e0b\u53ef\u80fd\u662f\u7537\u6027\u7684\u5b9d\u7238\uff1b\u53ea\u751f\u6210\u5019\u9009\uff0c\u9700\u72ec\u7acb\u8bc1\u636e\u786e\u8ba4\u3002"),
}
DOMAIN_SEMANTIC_REVIEW_ONLY_VARIANTS = frozenset(
    {
        "\u5408\u4fdd\u4eba\u5458",
        "\u805a\u5b9d",
        "\u5305\u4f53",
        "\u505c\u7626",
        "\u8ddf\u79d1\u6280\u8bb2",
        "\u4f4f\u4f60\u68c0",
        "\u84dd\u5c3e\u708e",
        "\u805a\u8d54",
        "\u5174\u8d77\u70b9",
        "\u7537\u6027\u7684\u5b9d\u5b9d",
    }
)




def _normalise_draft_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _canonical_key_pattern(key: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in re.split(r"\s+", key.strip()) if part]
    body = r"\s+".join(parts)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _find_canonical_key_text(text: str, key: str) -> str:
    if not text:
        return ""
    match = _canonical_key_pattern(key).search(text)
    return match.group(0) if match else ""


def _candidate_search_text(candidate: dict[str, Any]) -> str:
    parts = []
    for field in ("original_text", "candidate_text", "suggested_text", "context_text"):
        value = str(candidate.get(field) or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def _candidate_draft_correction_matches(candidate: dict[str, Any]) -> list[tuple[str, str, tuple[str, float, str]]]:
    kind = str(candidate.get("correction_type") or "ordinary_word").strip()
    if kind not in {"proper_noun", "term", "concept", "ordinary_word"}:
        return []
    if bool(candidate.get("needs_human_review")) or str(candidate.get("risk_level") or "") == "high":
        return []
    original = str(candidate.get("original_text") or "").strip()
    original_key = _normalise_draft_key(original)
    search_text = "\n".join(str(candidate.get(field) or "").strip() for field in ("original_text", "candidate_text", "suggested_text") if str(candidate.get(field) or "").strip())
    rows: list[tuple[str, str, tuple[str, float, str]]] = []
    seen: set[str] = set()
    for key, mapped in DRAFT_CANONICAL_CORRECTIONS.items():
        match_text = original if original_key == key else _find_canonical_key_text(search_text, key)
        if not match_text:
            continue
        dedupe = _normalise_draft_key(match_text) + "=>" + mapped[0].lower()
        if dedupe in seen:
            continue
        seen.add(dedupe)
        rows.append((match_text, key, mapped))
    return rows


def _candidate_draft_review_suggestions(candidate: dict[str, Any]) -> list[tuple[str, str, str]]:
    original = str(candidate.get("original_text") or "").strip()
    original_key = _normalise_draft_key(original)
    search_text = "\n".join(str(candidate.get(field) or "").strip() for field in ("original_text", "candidate_text", "suggested_text") if str(candidate.get(field) or "").strip())
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for key, suggestion in DRAFT_REVIEW_SUGGESTIONS.items():
        match_text = original if original_key == key else _find_canonical_key_text(search_text, key)
        if not match_text:
            continue
        dedupe = _normalise_draft_key(match_text)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        rows.append((match_text, key, suggestion))
    return rows


def _candidate_draft_generic_correction(candidate: dict[str, Any], *, min_confidence: float) -> dict[str, Any] | None:
    original = str(candidate.get("original_text") or "").strip()
    corrected = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    kind = str(candidate.get("correction_type") or "ordinary_word").strip()
    if not original or not corrected:
        return None
    discovered_by = str(candidate.get("discovered_by") or "")
    reason = str(candidate.get("reason") or "")
    source_types = {str(item) for item in (candidate.get("evidence_source_types") or []) if str(item)}
    if reason == "known_domain_semantic_suspect":
        return _candidate_draft_known_domain_semantic_correction(candidate, original=original, corrected=corrected, kind=kind, min_confidence=min_confidence)
    if "candidate_discovery" not in discovered_by and not (reason == "subtitle_text_differs_from_transcript" and source_types & {"platform_subtitle", "embedded_subtitle"}):
        return None
    if kind not in {"proper_noun", "term", "concept", "ordinary_word"}:
        return None
    if bool(candidate.get("needs_human_review")) or str(candidate.get("risk_level") or "") == "high":
        return None
    if _has_strong_source_opposition(candidate):
        return None
    if _normalize_compact(original) == _normalize_compact(corrected):
        return None
    if len(original) > 80 or len(corrected) > 80:
        return None
    if _fact_value_markers(original) or _fact_value_markers(corrected):
        return None
    if ACTION_HINT_RE.search(original) or ACTION_HINT_RE.search(corrected):
        return None
    evidence_ids = [str(item) for item in (candidate.get("evidence_ids") or []) if str(item).strip()]
    if not evidence_ids:
        return None
    strong_source_types = {
        "ocr",
        "structured_visual",
        "visual_understanding",
        "temporal_visual",
        "tagger",
        "platform_subtitle",
        "embedded_subtitle",
        "human_note",
    }
    if not (source_types & strong_source_types):
        return None
    discovery_confidence = _float(candidate.get("discovery_confidence"), 0.0)
    confidence = max(discovery_confidence, min_confidence, 0.93)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": "replace",
        "accept": True,
        "correction_type": kind,
        "original_text": original,
        "corrected_text": corrected,
        "confidence": round(float(confidence), 3),
        "semantic_rationale": "低风险 ASR/字幕疑似错词由 OCR/视觉/字幕/打标等强证据交叉支持，按通用语义纠错闭环写入纠正版 transcript。",
        "evidence_ids": evidence_ids,
        "human_confirmed": False,
        "needs_human_review": False,
        "safe_to_apply": True,
        "apply_scope": "segment",
    }


def _candidate_draft_known_domain_semantic_correction(candidate: dict[str, Any], *, original: str, corrected: str, kind: str, min_confidence: float) -> dict[str, Any] | None:
    if kind not in {"proper_noun", "term", "concept", "ordinary_word"}:
        return None
    if bool(candidate.get("needs_human_review")) or str(candidate.get("risk_level") or "") == "high":
        return None
    if _has_strong_source_opposition(candidate):
        return None
    if _normalize_compact(original) == _normalize_compact(corrected):
        return None
    if len(original) > 80 or len(corrected) > 80:
        return None
    evidence_ids = [str(item) for item in (candidate.get("evidence_ids") or []) if str(item).strip()]
    if not evidence_ids:
        return None
    confidence = max(_float(candidate.get("discovery_confidence"), 0.0), min_confidence, 0.93)
    rationale = str(candidate.get("domain_semantic_rationale") or "已知中文课程 ASR 易错词，且候选来自语义疑难点索引；本地 Codex 替代层按低风险通用语义纠错处理。")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": "replace",
        "accept": True,
        "correction_type": kind,
        "original_text": original,
        "corrected_text": corrected,
        "confidence": round(float(confidence), 3),
        "semantic_rationale": rationale,
        "evidence_ids": evidence_ids,
        "human_confirmed": False,
        "needs_human_review": False,
        "safe_to_apply": True,
        "apply_scope": "segment",
    }


def _candidate_draft_acronym_normalization(candidate: dict[str, Any], *, min_confidence: float) -> dict[str, Any] | None:
    original = str(candidate.get("original_text") or "").strip()
    corrected = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    kind = str(candidate.get("correction_type") or "proper_noun").strip()
    reason = str(candidate.get("reason") or "")
    if not original or not corrected:
        return None
    if kind not in {"proper_noun", "term", "concept", "ordinary_word"}:
        return None
    if reason not in {"odd_spaced_letters_or_acronym", "compound_spaced_tool_or_proper_noun"}:
        return None
    if bool(candidate.get("needs_human_review")) or str(candidate.get("risk_level") or "") == "high":
        return None
    if len(original) > 80 or len(corrected) > 80:
        return None
    if _fact_value_markers(original) or _fact_value_markers(corrected):
        return None
    if ACTION_HINT_RE.search(original) or ACTION_HINT_RE.search(corrected):
        return None
    if _normalize_compact(original) != _normalize_compact(corrected):
        return None
    if corrected not in SAFE_ACRONYM_NORMALIZATIONS:
        return None
    if original == corrected:
        return None
    if not (ODD_SPACING_RE.search(original) or COMPOUND_SPACED_ASCII_RE.search(original)):
        return None
    evidence_ids = [str(item) for item in (candidate.get("evidence_ids") or []) if str(item).strip()]
    if not evidence_ids:
        return None
    confidence = max(float(min_confidence), 0.91)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": "replace",
        "accept": True,
        "correction_type": "proper_noun" if kind == "ordinary_word" else kind,
        "original_text": original,
        "corrected_text": corrected,
        "confidence": round(float(confidence), 3),
        "semantic_rationale": "ASR 将英文缩写或复合专名按字母拆开，本次只做同字母、同顺序的大小写/空格规范化，不改变数字、事实值或动作语义。",
        "evidence_ids": evidence_ids,
        "human_confirmed": False,
        "needs_human_review": False,
        "safe_to_apply": True,
        "apply_scope": "all_segments",
    }

def _candidate_draft_number_correction(candidate: dict[str, Any], *, min_confidence: float) -> dict[str, Any] | None:
    original = str(candidate.get("original_text") or "").strip()
    corrected = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    if not original or not corrected:
        return None
    if str(candidate.get("correction_type") or "") != "number":
        return None
    discovered_by = str(candidate.get("discovered_by") or "")
    reason = str(candidate.get("reason") or "")
    if "candidate_discovery" not in discovered_by and reason not in {"visual_text_differs_from_transcript", "platform_subtitle_differs_from_transcript", "embedded_subtitle_differs_from_transcript"}:
        return None
    original_markers = set(_fact_value_markers(original))
    corrected_markers = set(_fact_value_markers(corrected))
    if not corrected_markers or corrected_markers == original_markers:
        return None
    evidence_ids = [str(item) for item in (candidate.get("evidence_ids") or []) if str(item).strip()]
    if not evidence_ids:
        return None
    discovery_confidence = _float(candidate.get("discovery_confidence"), 0.0)
    confidence = max(discovery_confidence, min_confidence, 0.96)
    if confidence < max(0.95, min_confidence):
        return None
    if not _has_strong_number_evidence(candidate, evidence_ids, corrected_text=corrected):
        return None
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": "replace",
        "accept": True,
        "correction_type": "number",
        "original_text": original,
        "corrected_text": corrected,
        "confidence": round(float(confidence), 3),
        "semantic_rationale": "数字/金额/年份类 ASR/字幕疑似错词由 OCR/字幕/人工等强证据直接支撑，并通过高风险数字校验闸门后写入纠正版 transcript。",
        "evidence_ids": evidence_ids,
        "human_confirmed": False,
        "needs_human_review": False,
        "safe_to_apply": True,
        "apply_scope": "segment",
    }


def _candidate_draft_action_correction(candidate: dict[str, Any], *, min_confidence: float) -> dict[str, Any] | None:
    if str(candidate.get("correction_type") or "") != "action":
        return None
    original_text = str(candidate.get("original_text") or "").strip()
    candidate_text = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    context_text = str(candidate.get("context_text") or original_text).strip()
    if not original_text or not candidate_text:
        return None
    discovered_by = str(candidate.get("discovered_by") or "")
    reason = str(candidate.get("reason") or "")
    if "candidate_discovery" not in discovered_by and reason not in {"action_or_step_word_in_transcript", "visual_text_differs_from_transcript", "tagger_text_differs_from_transcript"}:
        return None
    original_action, corrected_action = _candidate_action_replacement_terms(context_text, original_text, candidate_text)
    if not original_action or not corrected_action or original_action == corrected_action:
        return None
    evidence_ids = [str(item) for item in (candidate.get("evidence_ids") or []) if str(item).strip()]
    if not evidence_ids:
        return None
    discovery_confidence = _float(candidate.get("discovery_confidence"), 0.0)
    confidence = max(discovery_confidence, min_confidence, 0.94)
    if confidence < max(0.92, min_confidence):
        return None
    if not _has_strong_action_evidence(candidate, evidence_ids, corrected_text=corrected_action):
        return None
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": "replace",
        "accept": True,
        "correction_type": "action",
        "original_text": original_action,
        "corrected_text": corrected_action,
        "confidence": round(float(confidence), 3),
        "semantic_rationale": "动作/步骤类 ASR/字幕疑似错词由视觉/连续片段/OCR/打标/人工等强证据支撑，并通过动作高风险校验闸门后写入纠正版 transcript。",
        "evidence_ids": evidence_ids,
        "human_confirmed": False,
        "needs_human_review": False,
        "safe_to_apply": True,
        "apply_scope": "segment",
    }


def _candidate_action_replacement_terms(context_text: str, original_text: str, candidate_text: str) -> tuple[str, str]:
    original_actions = _action_markers(original_text) or _action_markers(context_text)
    candidate_actions = _action_markers(candidate_text)
    if not original_actions or not candidate_actions:
        return "", ""
    candidate_set = set(candidate_actions)
    original_set = set(original_actions)
    original_action = next((item for item in original_actions if item not in candidate_set), original_actions[0])
    corrected_action = next((item for item in candidate_actions if item not in original_set), candidate_actions[0])
    return original_action, corrected_action

FINAL_OUTPUT_CANDIDATES = [
    ("source_arbitrated_transcript_json", "source-arbitrated-transcript.json"),
    ("source_arbitrated_transcript_markdown", "source-arbitrated-transcript.md"),
    ("full_transcript", "exports/full-transcript.md"),
    ("full_body", "exports/full-body.md"),
    ("smart_summary", "exports/smart-summary.md"),
    ("smart_summary_codex", "exports/smart-summary.codex.md"),
    ("content_candidate_pack", "exports/content-candidate-pack.json"),
    ("content_material_card", "exports/content-material-card.json"),
]


def transcript_semantic_correction_output_contract(
    base_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the strict Broker contract shared by prompt and deep validation."""
    contract = dict(base_contract or {})
    required_keys = dict(contract.get("required_keys") or {})
    required_keys.update({"schema": "string", "source": "string", "decisions": "array"})
    nonempty_keys = list(contract.get("nonempty_keys") or [])
    for key in ("schema", "source", "decisions"):
        if key not in nonempty_keys:
            nonempty_keys.append(key)
    forbidden_markers = list(contract.get("forbidden_markers") or [])
    for marker in ("<think>", "```"):
        if marker not in forbidden_markers:
            forbidden_markers.append(marker)
    item_contracts = dict(contract.get("array_item_contracts") or {})
    item_contracts["decisions"] = {
        "required_keys": dict(STRICT_DECISION_REQUIRED_KEYS),
        "nonempty_keys": list(STRICT_DECISION_NONEMPTY_KEYS),
        "additional_keys_allowed": True,
    }
    return {
        **contract,
        "format": "json",
        "target": "content",
        "required_keys": required_keys,
        "nonempty_keys": nonempty_keys,
        "forbidden_markers": forbidden_markers,
        "array_item_contracts": item_contracts,
    }


def transcript_semantic_correction_model_instructions(base_instructions: str = "") -> str:
    """Append one canonical strict instruction block without duplicating it."""
    base = str(base_instructions or "").strip()
    if STRICT_MODEL_CONTRACT_MARKER in base:
        return base
    strict = (
        f"{STRICT_MODEL_CONTRACT_MARKER}. Return exactly one raw JSON object with "
        f"schema={RESULT_SCHEMA}, a non-empty source string, and a non-empty decisions array. "
        "Every decision must explicitly contain candidate_id, action, correction_type, "
        "original_text, corrected_text, confidence, rationale, evidence_ids, "
        "human_confirmed, and needs_human_review. candidate_id must exist in the supplied "
        "pack; original_text must be copied byte-for-byte from that candidate; evidence_ids "
        "must be a non-empty subset of that candidate's evidence_ids. Use action=replace only "
        "for an evidence-supported correction. Otherwise use action=needs_human_review or "
        "action=reject, retain the exact original_text, explain the reason, and still cite the "
        "relevant evidence_ids. Optimize for source fidelity: restore what the recording says, "
        "not whether a speaker's claim is externally true. Preserve attributed opinions and "
        "claims; external-world uncertainty alone is not a correction reason. Do not freely "
        "rewrite the transcript and do not emit Markdown fences, commentary, or hidden reasoning."
    )
    return f"{base}\n\n{strict}" if base else strict


def build_transcript_semantic_correction_pack(
    bundle_dir: str | Path,
    *,
    limit: int = 0,
    source_mode: str = "raw",
    write: bool = True,
    run_id: str | None = None,
    dependency_versions: dict | None = None,
) -> dict[str, Any]:
    if run_id:
        if limit or source_mode != "raw":
            raise ValueError("versioned_consumer_cannot_change_producer_options")
        from .versioned_semantic_run import consume
        return consume(bundle_dir, run_id, dependency_versions, "candidates")
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    timeline = _read_optional_json(root / "timeline.json")
    if not isinstance(timeline, list):
        timeline = []
    normalized_source_mode = str(source_mode or "raw").strip().lower()
    if normalized_source_mode not in {"raw", "canonical"}:
        raise ValueError(f"unsupported semantic correction source mode: {source_mode}")
    cues = (
        _load_best_cues(root, manifest)
        if normalized_source_mode == "canonical"
        else _load_raw_correction_cues(root, manifest, timeline)
    )
    sidecar_sources = _load_sidecar_sources(root, manifest, timeline)
    metadata_evidence = _metadata_evidence(root, manifest)
    chapters = _load_semantic_chapter_ranges(root)
    candidates = _build_candidates(
        cues,
        timeline,
        sidecar_sources=sidecar_sources,
        metadata_evidence=metadata_evidence,
        limit=limit,
    )
    candidates = _merge_imported_discovery_candidates(root, candidates, cues, timeline, sidecar_sources=sidecar_sources, metadata_evidence=metadata_evidence)
    _attach_candidate_context(candidates, cues, chapters=chapters)
    candidate_groups = _assign_candidate_groups(candidates)
    result = {
        "schema": PACK_SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "status": "pack_ready" if candidates else "no_candidates",
        "candidate_count": len(candidates),
        "candidate_group_count": len(candidate_groups),
        "limit": int(limit or 0),
        "source_mode": normalized_source_mode,
        "candidates": candidates,
        "candidate_groups": candidate_groups,
        "evidence_summary": _evidence_summary(candidates),
        "artifacts": _artifact_paths(root),
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "does_not_modify_raw_sources": True, "llm_or_codex_only_judges_candidates": True},
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_correction_pack", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-correction-pack.json", result)
            template = _result_template(result)
            write_json(root / "transcript-semantic-correction-result.template.json", template)
            (root / "transcript-semantic-correction-prompt.md").write_text(_render_prompt(result), encoding="utf-8")
            codex_path = root / "transcript-semantic-correction-result.codex.md"
            if not codex_path.exists():
                codex_path.write_text(_render_codex_stub(template), encoding="utf-8")
            _write_mcp_args(root, limit=limit)
            manifest.update({
                "transcript_semantic_correction_pack_json": "transcript-semantic-correction-pack.json",
                "transcript_semantic_correction_prompt_markdown": "transcript-semantic-correction-prompt.md",
                "transcript_semantic_correction_result_template_json": "transcript-semantic-correction-result.template.json",
                "transcript_semantic_correction_result_codex_markdown": "transcript-semantic-correction-result.codex.md",
                "mcp_transcript_semantic_correction_pack_args": "mcp-transcript-semantic-correction-pack.args.json",
                "mcp_transcript_semantic_correction_codex_draft_args": "mcp-transcript-semantic-correction-codex-draft.args.json",
                "mcp_transcript_semantic_correction_llm_draft_args": "mcp-transcript-semantic-correction-llm-draft.args.json",
                "mcp_validate_transcript_semantic_correction_args": "mcp-validate-transcript-semantic-correction.args.json",
                "mcp_transcript_semantic_correction_closure_args": "mcp-transcript-semantic-correction-closure.args.json",
                "mcp_transcript_semantic_correction_impact_report_args": "mcp-transcript-semantic-correction-impact-report.args.json",
                "mcp_transcript_semantic_readable_impact_report_args": "mcp-transcript-semantic-readable-impact-report.args.json",
                "mcp_transcript_semantic_correction_status_args": "mcp-transcript-semantic-correction-status.args.json",
                "transcript_semantic_correction_llm_prompt_markdown": "transcript-semantic-correction-llm-prompt.md",
                "transcript_semantic_correction_summary": {"status": result["status"], "candidate_count": len(candidates), "updated_at": result["updated_at"]},
            })
            write_json(root / "manifest.json", manifest)
    return result




def build_transcript_semantic_candidate_discovery_pack(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    limit: int = 40,
    cursor: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a Codex/LLM prompt for finding missed ASR/subtitle error candidates.

    This does not decide or apply corrections. It asks a model to propose extra
    candidate spans, then import_transcript_semantic_candidate_suggestions can
    merge those suggestions into the normal correction pack for validation.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = Path(input_json).expanduser().resolve() if input_json else root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    if not isinstance(pack, dict) or not isinstance(pack.get("candidates"), list):
        pack = build_transcript_semantic_correction_pack(root, limit=0, write=write)
        pack_path = root / "transcript-semantic-correction-pack.json"
    timeline = _read_optional_json(root / "timeline.json")
    if not isinstance(timeline, list):
        timeline = []
    cues = _load_raw_correction_cues(root, manifest, timeline)
    transcript_path = _raw_correction_source_path(root, manifest)
    if transcript_path is None and (root / "timeline.json").is_file():
        transcript_path = (root / "timeline.json").resolve()
    sidecar_sources = _load_sidecar_sources(root, manifest, timeline)
    metadata_evidence = _metadata_evidence(root, manifest)
    candidates = [row for row in (pack.get("candidates") or []) if isinstance(row, dict)]
    cursor_value = max(0, int(cursor or 0))
    requested_limit = int(limit or 0)
    effective_limit = min(
        DISCOVERY_MAX_SEGMENTS,
        requested_limit if requested_limit > 0 else DISCOVERY_MAX_SEGMENTS,
    )
    retained_indexes = [
        idx
        for idx, cue in enumerate(cues)
        if str(getattr(cue, "text", "") or "").strip()
    ]
    segments = _semantic_candidate_discovery_segments(
        cues,
        timeline,
        candidates,
        sidecar_sources=sidecar_sources,
        metadata_evidence=metadata_evidence,
        limit=effective_limit,
        cursor=cursor_value,
    )
    selected_source_indexes = [int(row["source_segment_index"]) for row in segments]
    last_selected_index = selected_source_indexes[-1] if selected_source_indexes else None
    remaining_indexes = [
        idx
        for idx in retained_indexes
        if idx > last_selected_index
    ] if last_selected_index is not None else [idx for idx in retained_indexes if idx >= cursor_value]
    complete = not remaining_indexes
    next_cursor = None if complete else (
        last_selected_index + 1 if last_selected_index is not None else remaining_indexes[0]
    )
    covered_count = len(
        retained_indexes
        if complete
        else [idx for idx in retained_indexes if idx < int(next_cursor or 0)]
    )
    coverage = {
        "retained_cue_count": len(retained_indexes),
        "selected_count": len(selected_source_indexes),
        "selected_source_indexes": selected_source_indexes,
        "covered_retained_cue_count": covered_count,
        "coverage_ratio": round(covered_count / len(retained_indexes), 6) if retained_indexes else 1.0,
    }
    template = _candidate_discovery_template(segments)
    scan_contract = {
        "cursor": cursor_value,
        "next_cursor": next_cursor,
        "complete": complete,
        "batch_limit": effective_limit,
        "selected_source_indexes": selected_source_indexes,
        "coverage": coverage,
    }
    prompt = _render_candidate_discovery_prompt(
        root=root,
        pack=pack,
        segments=segments,
        template=template,
        scan_contract=scan_contract,
    )
    result = {
        "schema": CANDIDATE_DISCOVERY_SCHEMA,
        "bundle_dir": str(root),
        "status": "discovery_prompt_ready" if segments else "no_segments_selected",
        "ok": True,
        "pack_json": str(pack_path),
        "segment_count": len(segments),
        "existing_candidate_count": len(candidates),
        "requested_limit": requested_limit,
        "limit": effective_limit,
        "cursor": cursor_value,
        "next_cursor": next_cursor,
        "complete": complete,
        "transcript_path": str(transcript_path) if transcript_path is not None else "",
        "transcript_sha256": _sha256_file(transcript_path) if transcript_path is not None else "",
        "selected_source_indexes": selected_source_indexes,
        "coverage": coverage,
        "segments": segments,
        "artifacts": _artifact_paths(root),
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "discovers_candidates_only": True, "does_not_apply_corrections": True, "validation_required_before_closure": True},
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_candidate_discovery_pack", timeout_seconds=1.0):
            discovery_pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
            write_json(discovery_pack_path, result)
            discovery_identity = _candidate_discovery_written_identity(
                discovery_pack_path,
                result,
            )
            template = {**template, "discovery_identity": discovery_identity}
            prompt = _render_candidate_discovery_prompt(
                root=root,
                pack=pack,
                segments=segments,
                template=template,
                scan_contract=scan_contract,
            )
            write_json(root / "transcript-semantic-candidate-discovery-template.json", template)
            (root / "transcript-semantic-candidate-discovery-prompt.md").write_text(prompt, encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-candidate-discovery-pack.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "limit": effective_limit, "cursor": cursor_value, "write": True})
            write_json(root / "mcp-import-transcript-semantic-candidate-suggestions.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-candidate-suggestions.codex.md"), "write": True})
            manifest.update({
                "transcript_semantic_candidate_discovery_pack_json": "transcript-semantic-candidate-discovery-pack.json",
                "transcript_semantic_candidate_discovery_prompt_markdown": "transcript-semantic-candidate-discovery-prompt.md",
                "transcript_semantic_candidate_discovery_template_json": "transcript-semantic-candidate-discovery-template.json",
                "mcp_transcript_semantic_candidate_discovery_pack_args": "mcp-transcript-semantic-candidate-discovery-pack.args.json",
                "mcp_import_transcript_semantic_candidate_suggestions_args": "mcp-import-transcript-semantic-candidate-suggestions.args.json",
                "transcript_semantic_candidate_discovery_summary": {"status": result["status"], "segment_count": len(segments), "updated_at": result["updated_at"]},
            })
            write_json(root / "manifest.json", manifest)
            result["discovery_identity"] = discovery_identity
    return result




def build_transcript_semantic_candidate_discovery_codex_draft(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    limit: int = 40,
    max_suggestions: int = 40,
    write: bool = True,
) -> dict[str, Any]:
    """Generate local Codex-substitute candidate suggestions from discovery pack.

    This is deliberately conservative: it proposes suspicious spans only. It
    does not validate, accept, close, or write corrected transcript sidecars.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = Path(input_json).expanduser().resolve() if input_json else root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    if not isinstance(pack, dict) or not isinstance(pack.get("candidates"), list):
        pack = build_transcript_semantic_correction_pack(root, limit=0, write=write)
        pack_path = root / "transcript-semantic-correction-pack.json"
    discovery = build_transcript_semantic_candidate_discovery_pack(root, input_json=pack_path, limit=limit, write=False)
    suggestions = _codex_candidate_discovery_suggestions(discovery, max_suggestions=max_suggestions)
    result_payload = {
        "schema": CANDIDATE_SUGGESTIONS_SCHEMA,
        "source": "local_codex_candidate_discovery_draft",
        "bundle_dir": str(root),
        "discovery_pack_json": str(root / "transcript-semantic-candidate-discovery-pack.json"),
        "suggestions": suggestions,
        "operator_boundary": {
            "local_codex_substitute": True,
            "no_cloud_call": True,
            "suggestions_only": True,
            "does_not_modify_raw_sources": True,
            "does_not_modify_correction_pack": True,
            "import_and_validation_required_before_closure": True,
        },
        "updated_at": now_iso(),
    }
    status = "codex_suggestions_ready" if suggestions else "no_safe_codex_suggestions"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_candidate_discovery_codex_draft.v1",
        "bundle_dir": str(root),
        "status": status,
        "ok": True,
        "pack_json": str(pack_path),
        "segment_count": int(discovery.get("segment_count") or 0),
        "suggestion_count": len(suggestions),
        "result_json": str(root / "transcript-semantic-candidate-suggestions.codex.json"),
        "result_markdown": str(root / "transcript-semantic-candidate-suggestions.codex.md"),
        "operator_boundary": result_payload["operator_boundary"],
        "updated_at": result_payload["updated_at"],
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_candidate_discovery_codex_draft", timeout_seconds=1.0):
            discovery_pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
            write_json(discovery_pack_path, discovery)
            discovery_identity = _candidate_discovery_written_identity(
                discovery_pack_path,
                discovery,
            )
            discovery_segments = [row for row in (discovery.get("segments") or []) if isinstance(row, dict)]
            discovery_template = {
                **_candidate_discovery_template(discovery_segments),
                "discovery_identity": discovery_identity,
            }
            result_payload["discovery_identity"] = discovery_identity
            write_json(root / "transcript-semantic-candidate-discovery-template.json", discovery_template)
            (root / "transcript-semantic-candidate-discovery-prompt.md").write_text(_render_candidate_discovery_prompt(root=root, pack=pack, segments=discovery_segments, template=discovery_template), encoding="utf-8")
            write_json(root / "transcript-semantic-candidate-suggestions.codex.json", result_payload)
            (root / "transcript-semantic-candidate-suggestions.codex.md").write_text(_render_codex_candidate_suggestions_markdown(result_payload), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "limit": int(limit or 0), "max_suggestions": int(max_suggestions or 0), "write": True})
            write_json(root / "mcp-import-transcript-semantic-candidate-suggestions.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-candidate-suggestions.codex.md"), "write": True})
            manifest.update({
                "transcript_semantic_candidate_discovery_pack_json": "transcript-semantic-candidate-discovery-pack.json",
                "transcript_semantic_candidate_discovery_prompt_markdown": "transcript-semantic-candidate-discovery-prompt.md",
                "transcript_semantic_candidate_discovery_template_json": "transcript-semantic-candidate-discovery-template.json",
                "transcript_semantic_candidate_suggestions_codex_json": "transcript-semantic-candidate-suggestions.codex.json",
                "transcript_semantic_candidate_suggestions_codex_markdown": "transcript-semantic-candidate-suggestions.codex.md",
                "mcp_transcript_semantic_candidate_discovery_codex_draft_args": "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json",
                "mcp_import_transcript_semantic_candidate_suggestions_args": "mcp-import-transcript-semantic-candidate-suggestions.args.json",
                "transcript_semantic_candidate_discovery_codex_draft_summary": {"status": status, "segment_count": int(discovery.get("segment_count") or 0), "suggestion_count": len(suggestions), "updated_at": result["updated_at"]},
            })
            write_json(root / "manifest.json", manifest)
    return result
def build_transcript_semantic_candidate_discovery_llm_draft(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    limit: int = 40,
    write: bool = True,
) -> dict[str, Any]:
    """Plan or execute text LLM discovery of missed semantic correction candidates.

    This is preview-first. Even when execute=True, model output is saved as
    candidate suggestions only; import_transcript_semantic_candidate_suggestions
    is still required before the normal validate/closure pipeline can act.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = Path(input_json).expanduser().resolve() if input_json else root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    if not isinstance(pack, dict) or not isinstance(pack.get("candidates"), list):
        pack = build_transcript_semantic_correction_pack(root, limit=0, write=write)
        pack_path = root / "transcript-semantic-correction-pack.json"
    discovery = build_transcript_semantic_candidate_discovery_pack(root, input_json=pack_path, limit=limit, write=False)
    segments = [row for row in (discovery.get("segments") or []) if isinstance(row, dict)]
    template = _candidate_discovery_template(segments)
    prompt = _render_candidate_discovery_prompt(root=root, pack=pack, segments=segments, template=template)
    cfg = resolve_text_provider_config(provider_config or {})
    provider_public = _public_text_provider_config(cfg)
    status = "planned"
    error = ""
    raw_model_output = ""
    result_payload: dict[str, Any] | None = None
    if execute:
        if not segments:
            status = "no_segments_selected"
        else:
            response = model_task_api_call(
                "transcript_candidate_discovery", provider_config=cfg,
                messages=[{"role": "user", "content": prompt}], execute=True, temperature=0,
                response_format={"type": "json_object"}, max_tokens=4096, write=False,
            )
            raw_model_output = str(response.get("content") or "")
            if not response.get("ok"):
                status = str(response.get("error") or "provider_failed")
                error = str(response.get("error") or "provider_failed")
            else:
                try:
                    parsed = extract_json_document(raw_model_output, require_object=True)
                    result_payload = _normalise_candidate_suggestions_payload(parsed if isinstance(parsed, dict) else {}, discovery_pack=discovery)
                    status = "executed"
                except Exception as exc:
                    status = "model_output_parse_failed"
                    error = str(exc)
    if result_payload is None:
        result_payload = {
            "schema": CANDIDATE_SUGGESTIONS_SCHEMA,
            "source": "text_llm_candidate_discovery" if execute else "text_llm_candidate_discovery_plan",
            "bundle_dir": str(root),
            "discovery_pack_json": str(root / "transcript-semantic-candidate-discovery-pack.json"),
            "suggestions": [],
            "operator_boundary": {
                "preview_by_default": True,
                "execute_may_call_text_llm": True,
                "suggestions_only": True,
                "does_not_modify_raw_sources": True,
                "does_not_modify_correction_pack": True,
                "import_and_validation_required_before_closure": True,
            },
            "updated_at": now_iso(),
        }
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_candidate_discovery_llm_draft.v1",
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"planned", "executed", "no_segments_selected"},
        "execute": bool(execute),
        "pack_json": str(pack_path),
        "segment_count": len(segments),
        "suggestion_count": len(result_payload.get("suggestions") or []),
        "provider": provider_public,
        "prompt_markdown": str(root / "transcript-semantic-candidate-discovery-llm-prompt.md"),
        "result_json": str(root / "transcript-semantic-candidate-suggestions.llm.json"),
        "result_markdown": str(root / "transcript-semantic-candidate-suggestions.llm.md"),
        "error": error,
        "raw_model_output_saved": bool(execute and raw_model_output and status == "model_output_parse_failed"),
        "operator_boundary": result_payload.get("operator_boundary", {}),
        "updated_at": result_payload.get("updated_at") or now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_candidate_discovery_llm_draft", timeout_seconds=1.0):
            discovery_pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
            write_json(discovery_pack_path, discovery)
            discovery_identity = _candidate_discovery_written_identity(
                discovery_pack_path,
                discovery,
            )
            template = {**template, "discovery_identity": discovery_identity}
            result_payload["discovery_identity"] = discovery_identity
            prompt = _render_candidate_discovery_prompt(root=root, pack=pack, segments=segments, template=template)
            write_json(root / "transcript-semantic-candidate-discovery-template.json", template)
            (root / "transcript-semantic-candidate-discovery-prompt.md").write_text(_render_candidate_discovery_prompt(root=root, pack=pack, segments=segments, template=template), encoding="utf-8")
            (root / "transcript-semantic-candidate-discovery-llm-prompt.md").write_text(prompt, encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "provider_config": {}, "execute": False, "limit": int(limit or 0), "write": True})
            if execute and raw_model_output and status == "model_output_parse_failed":
                (root / "transcript-semantic-candidate-suggestions.llm.raw.txt").write_text(raw_model_output, encoding="utf-8")
                manifest["transcript_semantic_candidate_suggestions_llm_raw_text"] = "transcript-semantic-candidate-suggestions.llm.raw.txt"
            if execute and status == "executed":
                write_json(root / "transcript-semantic-candidate-suggestions.llm.json", result_payload)
                (root / "transcript-semantic-candidate-suggestions.llm.md").write_text(_render_candidate_suggestions_markdown(result_payload), encoding="utf-8")
                manifest["transcript_semantic_candidate_suggestions_llm_json"] = "transcript-semantic-candidate-suggestions.llm.json"
                manifest["transcript_semantic_candidate_suggestions_llm_markdown"] = "transcript-semantic-candidate-suggestions.llm.md"
            manifest["transcript_semantic_candidate_discovery_llm_prompt_markdown"] = "transcript-semantic-candidate-discovery-llm-prompt.md"
            manifest["transcript_semantic_candidate_discovery_llm_draft_summary"] = {"status": status, "segment_count": len(segments), "suggestion_count": result["suggestion_count"], "updated_at": result["updated_at"]}
            manifest["mcp_transcript_semantic_candidate_discovery_llm_draft_args"] = "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json"
            write_json(root / "manifest.json", manifest)
    return result


def _candidate_discovery_written_identity(
    discovery_pack_path: Path,
    discovery_pack: dict[str, Any],
) -> dict[str, Any]:
    return {
        "transcript_sha256": str(discovery_pack.get("transcript_sha256") or ""),
        "discovery_pack_sha256": _sha256_file(discovery_pack_path),
        "cursor": max(0, int(discovery_pack.get("cursor") or 0)),
        "selected_source_indexes": [
            int(value)
            for value in discovery_pack.get("selected_source_indexes") or []
            if str(value).lstrip("-").isdigit()
        ],
    }


def _candidate_discovery_import_identity_issues(
    root: Path,
    discovery_pack: dict[str, Any],
    *,
    discovery_pack_sha256: str,
    identity: dict[str, Any],
    suggestions: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    required = (
        "transcript_sha256",
        "discovery_pack_sha256",
        "cursor",
        "selected_source_indexes",
    )
    for key in required:
        if key not in identity:
            issues.append(f"identity_{key}_missing")

    recorded_transcript_sha256 = str(discovery_pack.get("transcript_sha256") or "")
    identity_transcript_sha256 = str(identity.get("transcript_sha256") or "")
    if identity_transcript_sha256 and identity_transcript_sha256 != recorded_transcript_sha256:
        issues.append("identity_transcript_sha256_mismatch")
    if str(identity.get("discovery_pack_sha256") or "") != discovery_pack_sha256:
        issues.append("identity_discovery_pack_sha256_mismatch")
    try:
        identity_cursor = int(identity.get("cursor"))
    except (TypeError, ValueError):
        identity_cursor = -1
    if identity_cursor != max(0, int(discovery_pack.get("cursor") or 0)):
        issues.append("identity_cursor_mismatch")

    expected_indexes = [
        int(value)
        for value in discovery_pack.get("selected_source_indexes") or []
        if str(value).lstrip("-").isdigit()
    ]
    identity_indexes = identity.get("selected_source_indexes")
    if not isinstance(identity_indexes, list):
        identity_indexes = []
    normalized_identity_indexes = [
        int(value)
        for value in identity_indexes
        if str(value).lstrip("-").isdigit()
    ]
    if normalized_identity_indexes != expected_indexes or len(normalized_identity_indexes) != len(identity_indexes):
        issues.append("identity_selected_source_indexes_mismatch")

    transcript_value = str(discovery_pack.get("transcript_path") or "").strip()
    transcript_path = Path(transcript_value).expanduser() if transcript_value else None
    if transcript_path is not None and not transcript_path.is_absolute():
        transcript_path = root / transcript_path
    try:
        transcript_path = transcript_path.resolve() if transcript_path is not None else None
    except OSError:
        transcript_path = None
    if transcript_path is None:
        issues.append("transcript_path_missing")
    elif not transcript_path.is_relative_to(root):
        issues.append("transcript_path_outside_bundle")
    elif not transcript_path.is_file():
        issues.append("transcript_path_missing")
    else:
        current_transcript_sha256 = _sha256_file(transcript_path)
        if current_transcript_sha256 != recorded_transcript_sha256:
            issues.append("current_transcript_sha256_mismatch")

    selected = set(expected_indexes)
    for row_number, suggestion in enumerate(suggestions, start=1):
        raw_index = suggestion.get("source_segment_index")
        if not str(raw_index).lstrip("-").isdigit():
            issues.append(f"suggestion_{row_number}_source_segment_index_missing")
            continue
        if int(raw_index) not in selected:
            issues.append(f"suggestion_{row_number}_source_segment_index_outside_page")
    return list(dict.fromkeys(issues))


def _candidate_discovery_progress_preflight(
    root: Path,
    *,
    identity: dict[str, Any],
    discovery_pack_sha256: str,
    suggestions_input_sha256: str,
    source_pack_sha256: str,
) -> tuple[str, list[str], dict[str, Any]]:
    receipt_path = root / DISCOVERY_SCAN_RECEIPT_NAME
    receipt = _read_optional_json(receipt_path)
    cursor = max(0, int(identity.get("cursor") or 0))
    if not isinstance(receipt, dict) or not receipt:
        return (
            "ready" if cursor == 0 else "rejected",
            [] if cursor == 0 else ["missing_prior_page_receipt"],
            {},
        )

    issues: list[str] = []
    if str(receipt.get("schema") or "") != DISCOVERY_SCAN_RECEIPT_SCHEMA:
        issues.append("scan_receipt_schema_mismatch")
    if str(receipt.get("transcript_sha256") or "") != str(identity.get("transcript_sha256") or ""):
        issues.append("scan_receipt_transcript_sha256_mismatch")
    if str(receipt.get("source_pack_sha256") or "") != source_pack_sha256:
        issues.append("scan_receipt_source_pack_sha256_mismatch")
    batches = [row for row in receipt.get("batches") or [] if isinstance(row, dict)]
    final_batch = batches[-1] if batches else {}
    if final_batch and str(final_batch.get("discovery_pack_sha256") or "") == discovery_pack_sha256:
        if str(final_batch.get("suggestions_input_sha256") or "") == suggestions_input_sha256:
            return "already_recorded", issues, receipt
        issues.append("page_already_committed_with_different_result")
        return "rejected", issues, receipt
    if receipt.get("complete") is True or receipt.get("next_cursor") is None:
        issues.append("scan_already_complete")
    else:
        try:
            expected_cursor = int(receipt.get("next_cursor"))
        except (TypeError, ValueError):
            expected_cursor = -1
        if cursor != expected_cursor:
            issues.append("page_cursor_not_sequential")
    return ("rejected" if issues else "ready"), issues, receipt


def _candidate_discovery_scan_receipt(
    *,
    prior: dict[str, Any],
    discovery_pack: dict[str, Any],
    discovery_pack_sha256: str,
    suggestions_input_sha256: str,
    source_pack_sha256: str,
    suggestion_count: int,
    imported_candidate_count: int,
    skipped_count: int,
    updated_at: str,
) -> dict[str, Any]:
    coverage = discovery_pack.get("coverage") if isinstance(discovery_pack.get("coverage"), dict) else {}
    retained_count = max(0, int(coverage.get("retained_cue_count") or 0))
    covered_count = max(0, int(coverage.get("covered_retained_cue_count") or 0))
    complete = bool(
        discovery_pack.get("complete") is True
        and discovery_pack.get("next_cursor") is None
        and covered_count >= retained_count
    )
    batch = {
        "cursor": max(0, int(discovery_pack.get("cursor") or 0)),
        "next_cursor": None if complete else discovery_pack.get("next_cursor"),
        "complete": complete,
        "selected_source_indexes": [
            int(value)
            for value in discovery_pack.get("selected_source_indexes") or []
            if str(value).lstrip("-").isdigit()
        ],
        "discovery_pack_sha256": discovery_pack_sha256,
        "suggestions_input_sha256": suggestions_input_sha256,
        "suggestion_count": int(suggestion_count),
        "imported_candidate_count": int(imported_candidate_count),
        "skipped_count": int(skipped_count),
        "imported_at": updated_at,
    }
    batches = [row for row in prior.get("batches") or [] if isinstance(row, dict)]
    batches.append(batch)
    return {
        "schema": DISCOVERY_SCAN_RECEIPT_SCHEMA,
        "status": "completed" if complete else "in_progress",
        "complete": complete,
        "cursor": batch["cursor"],
        "next_cursor": batch["next_cursor"],
        "transcript_sha256": str(discovery_pack.get("transcript_sha256") or ""),
        "source_pack_sha256": source_pack_sha256,
        "discovery_pack_sha256": discovery_pack_sha256,
        "suggestions_input_sha256": suggestions_input_sha256,
        "coverage": coverage,
        "batches": batches[-256:],
        "updated_at": updated_at,
    }


def import_transcript_semantic_candidate_suggestions(
    bundle_dir: str | Path,
    *,
    input_json: str | Path,
    write: bool = True,
) -> dict[str, Any]:
    """Merge Codex/LLM-discovered suspicious spans into the normal pack.

    Imported suggestions are candidates only. They still require normal review,
    validation, and closure before any corrected transcript is written.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    input_path = Path(input_json).expanduser().resolve()
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    if not isinstance(pack, dict) or not isinstance(pack.get("candidates"), list):
        pack = build_transcript_semantic_correction_pack(root, limit=0, write=write)
    timeline = _read_optional_json(root / "timeline.json")
    if not isinstance(timeline, list):
        timeline = []
    cues = _load_raw_correction_cues(root, manifest, timeline)
    sidecar_sources = _load_sidecar_sources(root, manifest, timeline)
    metadata_evidence = _metadata_evidence(root, manifest)
    payload = _load_import(input_path)
    suggestions = payload.get("suggestions") if isinstance(payload.get("suggestions"), list) else payload.get("candidates")
    suggestions = [row for row in (suggestions or []) if isinstance(row, dict)]
    identity = payload.get("discovery_identity") if isinstance(payload.get("discovery_identity"), dict) else {}
    identity_bound = bool(identity)
    discovery_pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
    discovery_pack = _read_optional_json(discovery_pack_path)
    discovery_pack_sha256 = _sha256_file(discovery_pack_path) if discovery_pack_path.is_file() else ""
    suggestions_input_sha256 = _sha256_file(input_path)
    source_pack_sha256 = _sha256_file(pack_path) if pack_path.is_file() else ""
    progress_issues: list[str] = []
    progress_preflight = "legacy"
    prior_receipt: dict[str, Any] = {}
    if identity_bound:
        if not isinstance(discovery_pack, dict) or not discovery_pack:
            progress_issues.append("discovery_pack_missing_or_invalid")
        else:
            progress_issues.extend(
                _candidate_discovery_import_identity_issues(
                    root,
                    discovery_pack,
                    discovery_pack_sha256=discovery_pack_sha256,
                    identity=identity,
                    suggestions=suggestions,
                )
            )
        if not progress_issues:
            progress_preflight, receipt_issues, prior_receipt = _candidate_discovery_progress_preflight(
                root,
                identity=identity,
                discovery_pack_sha256=discovery_pack_sha256,
                suggestions_input_sha256=suggestions_input_sha256,
                source_pack_sha256=source_pack_sha256,
            )
            progress_issues.extend(receipt_issues)

    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions_import.v1",
        "bundle_dir": str(root),
        "status": "pending",
        "ok": True,
        "input_json": str(input_path),
        "suggestion_count": len(suggestions),
        "imported_candidate_count": 0,
        "skipped_count": 0,
        "imported_candidate_ids": [],
        "skipped": [],
        "pack_json": str(pack_path),
        "progress_recorded": False,
        "progress_status": "identity_missing" if not identity_bound else "ready_to_record",
        "progress_issues": progress_issues,
        "progress_receipt": str(root / DISCOVERY_SCAN_RECEIPT_NAME),
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "imports_candidates_only": True, "does_not_apply_corrections": True, "validation_required_before_closure": True},
        "updated_at": now_iso(),
    }
    if identity_bound and (progress_issues or progress_preflight == "rejected"):
        result.update({"status": "progress_identity_rejected", "ok": False})
        return result
    if identity_bound and progress_preflight == "already_recorded":
        result.update(
            {
                "status": "already_imported",
                "progress_recorded": True,
                "progress_status": "already_recorded",
            }
        )
        return result

    candidates = [dict(row) for row in (pack.get("candidates") or []) if isinstance(row, dict)]
    seen = {(str(row.get("original_text") or "").lower(), str(row.get("candidate_text") or row.get("suggested_text") or "").lower(), str(row.get("correction_type") or "ordinary_word")) for row in candidates}
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, suggestion in enumerate(suggestions, start=1):
        candidate, reason = _candidate_from_discovery_suggestion(suggestion, cues, timeline, sidecar_sources=sidecar_sources, metadata_evidence=metadata_evidence)
        if candidate is None:
            skipped.append({"row_number": idx, "reason": reason, "suggestion": suggestion})
            continue
        key = (str(candidate.get("original_text") or "").lower(), str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").lower(), str(candidate.get("correction_type") or "ordinary_word"))
        if key in seen:
            skipped.append({"row_number": idx, "reason": "duplicate_candidate", "suggestion": suggestion})
            continue
        seen.add(key)
        candidate["candidate_id"] = f"semcorr-{len(candidates)+1:04d}"
        candidate["discovered_by"] = str(payload.get("source") or "codex_or_llm_candidate_discovery")
        candidates.append(candidate)
        imported.append(candidate)
    invalid_skips = [row for row in skipped if str(row.get("reason") or "") != "duplicate_candidate"]
    if identity_bound and invalid_skips:
        result.update(
            {
                "status": "progress_identity_rejected",
                "ok": False,
                "skipped_count": len(skipped),
                "skipped": skipped,
                "progress_issues": [
                    f"suggestion_invalid:{str(row.get('reason') or 'unknown')}"
                    for row in invalid_skips
                ],
            }
        )
        return result

    candidate_groups = _assign_candidate_groups(candidates)
    pack.update({"status": "pack_ready" if candidates else "no_candidates", "candidate_count": len(candidates), "candidate_group_count": len(candidate_groups), "candidates": candidates, "candidate_groups": candidate_groups, "evidence_summary": _evidence_summary(candidates), "updated_at": now_iso()})
    result.update(
        {
            "status": "imported" if imported else "no_candidates_imported",
            "imported_candidate_count": len(imported),
            "skipped_count": len(skipped),
            "imported_candidate_ids": [row.get("candidate_id") for row in imported],
            "skipped": skipped,
        }
    )
    if not write:
        result["progress_status"] = "preview_only" if identity_bound else "identity_missing"
        return result

    with bundle_write_lock(root, operation="import_transcript_semantic_candidate_suggestions", timeout_seconds=1.0):
        if identity_bound:
            locked_discovery_pack = _read_optional_json(discovery_pack_path)
            locked_discovery_sha256 = _sha256_file(discovery_pack_path) if discovery_pack_path.is_file() else ""
            locked_source_sha256 = _sha256_file(pack_path) if pack_path.is_file() else ""
            locked_issues = []
            if locked_source_sha256 != source_pack_sha256:
                locked_issues.append("source_pack_sha256_drift_during_import")
            if not isinstance(locked_discovery_pack, dict) or not locked_discovery_pack:
                locked_issues.append("discovery_pack_missing_or_invalid")
            else:
                locked_issues.extend(
                    _candidate_discovery_import_identity_issues(
                        root,
                        locked_discovery_pack,
                        discovery_pack_sha256=locked_discovery_sha256,
                        identity=identity,
                        suggestions=suggestions,
                    )
                )
            locked_preflight, locked_receipt_issues, locked_prior_receipt = _candidate_discovery_progress_preflight(
                root,
                identity=identity,
                discovery_pack_sha256=locked_discovery_sha256,
                suggestions_input_sha256=suggestions_input_sha256,
                source_pack_sha256=locked_source_sha256,
            )
            locked_issues.extend(locked_receipt_issues)
            if locked_preflight == "already_recorded" and not locked_issues:
                result.update({"status": "already_imported", "progress_recorded": True, "progress_status": "already_recorded"})
                return result
            if locked_preflight == "rejected" or locked_issues:
                result.update({"status": "progress_identity_rejected", "ok": False, "progress_issues": list(dict.fromkeys(locked_issues))})
                return result
            discovery_pack = locked_discovery_pack
            discovery_pack_sha256 = locked_discovery_sha256
            prior_receipt = locked_prior_receipt

        persisted_path = root / "transcript-semantic-candidate-suggestions-imported.json"
        persisted = _read_optional_json(persisted_path)
        persisted_rows = persisted.get("imported_candidates") if isinstance(persisted, dict) else []
        if not isinstance(persisted_rows, list):
            persisted_rows = []
        persisted_keys = set()
        for persisted_row in persisted_rows:
            if not isinstance(persisted_row, dict):
                continue
            persisted_candidate = persisted_row.get("candidate") if isinstance(persisted_row.get("candidate"), dict) else persisted_row
            persisted_keys.add((
                str(persisted_candidate.get("original_text") or "").lower(),
                str(persisted_candidate.get("candidate_text") or persisted_candidate.get("suggested_text") or "").lower(),
                str(persisted_candidate.get("correction_type") or "ordinary_word"),
            ))
        for row in imported:
            key = (
                str(row.get("original_text") or "").lower(),
                str(row.get("candidate_text") or row.get("suggested_text") or "").lower(),
                str(row.get("correction_type") or "ordinary_word"),
            )
            if key in persisted_keys:
                continue
            persisted_keys.add(key)
            persisted_rows.append({"candidate": row, "input_json": str(input_path), "imported_at": result["updated_at"]})
        write_json(
            persisted_path,
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions_imported.v1",
                "bundle_dir": str(root),
                "imported_candidate_count": len(persisted_rows),
                "imported_candidates": persisted_rows,
                "updated_at": result["updated_at"],
            },
        )
        if imported:
            write_json(pack_path, pack)
            write_json(root / "transcript-semantic-correction-result.template.json", _result_template(pack))
            (root / "transcript-semantic-correction-prompt.md").write_text(_render_prompt(pack), encoding="utf-8")
            _write_mcp_args(root, limit=int(pack.get("limit") or 0))
        else:
            # A completed scan with no new candidates is progress evidence,
            # not a new semantic input revision. Preserve the admitted pack's
            # exact bytes so its existing validation/closure remain valid.
            pack = _read_optional_json(pack_path)

        if identity_bound:
            receipt = _candidate_discovery_scan_receipt(
                prior=prior_receipt,
                discovery_pack=discovery_pack,
                discovery_pack_sha256=discovery_pack_sha256,
                suggestions_input_sha256=suggestions_input_sha256,
                source_pack_sha256=_sha256_file(pack_path),
                suggestion_count=len(suggestions),
                imported_candidate_count=len(imported),
                skipped_count=len(skipped),
                updated_at=result["updated_at"],
            )
            result.update(
                {
                    "progress_recorded": True,
                    "progress_status": str(receipt["status"]),
                    "next_cursor": receipt["next_cursor"],
                }
            )
        write_json(root / "transcript-semantic-candidate-suggestions-import.json", result)
        if identity_bound:
            # This is the page commit marker. Everything needed to retry an
            # interrupted import has already been written before this point.
            write_json(root / DISCOVERY_SCAN_RECEIPT_NAME, receipt)
        manifest = _read_manifest(root)
        manifest["transcript_semantic_candidate_suggestions_import_json"] = "transcript-semantic-candidate-suggestions-import.json"
        manifest["transcript_semantic_candidate_suggestions_imported_json"] = "transcript-semantic-candidate-suggestions-imported.json"
        if identity_bound:
            manifest["transcript_semantic_candidate_discovery_scan_receipt_json"] = DISCOVERY_SCAN_RECEIPT_NAME
        manifest["transcript_semantic_correction_summary"] = {"status": pack["status"], "candidate_count": len(candidates), "updated_at": pack["updated_at"]}
        write_json(root / "manifest.json", manifest)
    return result
def build_transcript_semantic_correction_codex_draft(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    min_confidence: float = 0.88,
    write: bool = True,
    run_id: str | None = None,
    dependency_versions: dict | None = None,
) -> dict[str, Any]:
    """Build a conservative local Codex-substitute correction result.

    This is not a general language model. It only accepts very obvious ASR
    spelling/spacing errors that match a small built-in canonical map. Other
    candidates stay for Codex/LLM/human review.
    """
    if run_id:
        if input_json is not None or min_confidence != 0.88:
            raise ValueError("versioned_consumer_cannot_change_producer_options")
        from .versioned_semantic_run import consume
        return consume(bundle_dir, run_id, dependency_versions, "local_draft")
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = Path(input_json).expanduser().resolve() if input_json else root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    candidates = [row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    decisions: list[dict[str, Any]] = []
    review_suggestions: list[dict[str, Any]] = []
    seen_replacements: set[tuple[str, str]] = set()
    seen_suggestions: set[tuple[str, str]] = set()
    for candidate in candidates:
        for original, key, mapped in _candidate_draft_correction_matches(candidate):
            corrected, confidence, rationale = mapped
            if confidence < min_confidence or corrected == original:
                continue
            dedupe_key = (key, corrected.lower())
            if dedupe_key in seen_replacements:
                continue
            seen_replacements.add(dedupe_key)
            decisions.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "action": "replace",
                    "accept": True,
                    "correction_type": "proper_noun",
                    "original_text": original,
                    "corrected_text": corrected,
                    "confidence": confidence,
                    "semantic_rationale": rationale,
                    "evidence_ids": candidate.get("evidence_ids") or [],
                    "human_confirmed": False,
                    "needs_human_review": False,
                    "safe_to_apply": True,
                    "apply_scope": "all_segments",
                }
            )
        acronym_decision = _candidate_draft_acronym_normalization(candidate, min_confidence=min_confidence)
        if acronym_decision:
            dedupe_key = (_normalise_draft_key(str(acronym_decision.get("original_text") or "")), str(acronym_decision.get("corrected_text") or "").lower())
            if dedupe_key not in seen_replacements:
                seen_replacements.add(dedupe_key)
                decisions.append(acronym_decision)
        generic_decision = _candidate_draft_generic_correction(candidate, min_confidence=min_confidence)
        if generic_decision:
            dedupe_key = (_normalise_draft_key(str(generic_decision.get("original_text") or "")), str(generic_decision.get("corrected_text") or "").lower())
            if dedupe_key not in seen_replacements:
                seen_replacements.add(dedupe_key)
                decisions.append(generic_decision)
        number_decision = _candidate_draft_number_correction(candidate, min_confidence=min_confidence)
        if number_decision:
            dedupe_key = (_normalise_draft_key(str(number_decision.get("original_text") or "")), str(number_decision.get("corrected_text") or "").lower())
            if dedupe_key not in seen_replacements:
                seen_replacements.add(dedupe_key)
                decisions.append(number_decision)
        action_decision = _candidate_draft_action_correction(candidate, min_confidence=min_confidence)
        if action_decision:
            dedupe_key = (_normalise_draft_key(str(action_decision.get("original_text") or "")), str(action_decision.get("corrected_text") or "").lower())
            if dedupe_key not in seen_replacements:
                seen_replacements.add(dedupe_key)
                decisions.append(action_decision)
        for original, key, suggestion in _candidate_draft_review_suggestions(candidate):
            dedupe_key = (key, _normalise_draft_key(original))
            if dedupe_key in seen_suggestions:
                continue
            seen_suggestions.add(dedupe_key)
            review_suggestions.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "original_text": original,
                    "time_range": candidate.get("time_range"),
                    "suggestion": suggestion,
                }
            )
    result_payload = {
        "schema": RESULT_SCHEMA,
        "source": "codex_substitute_local_draft",
        "pack_json": str(pack_path),
        "min_confidence": float(min_confidence),
        "decisions": decisions,
        "review_suggestions": review_suggestions,
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "conservative_known_terms_only": True,
            "imports_low_risk_high_confidence_candidates": True,
            "imports_strong_evidence_number_candidates": True,
            "imports_strong_evidence_action_candidates": True,
            "imports_safe_acronym_normalizations": True,
            "does_not_modify_raw_sources": True,
        },
        "updated_at": now_iso(),
    }
    status = "draft_ready" if decisions else "no_safe_draft_decisions"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_codex_draft.v1",
        "bundle_dir": str(root),
        "status": status,
        "ok": bool(decisions),
        "decision_count": len(decisions),
        "review_suggestion_count": len(review_suggestions),
        "result_json": str(root / "transcript-semantic-correction-result.codex.json"),
        "result_markdown": str(root / "transcript-semantic-correction-result.codex.md"),
        "decisions": decisions,
        "review_suggestions": review_suggestions,
        "operator_boundary": result_payload["operator_boundary"],
        "updated_at": result_payload["updated_at"],
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_correction_codex_draft", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-correction-result.codex.json", result_payload)
            (root / "transcript-semantic-correction-result.codex.md").write_text(_render_codex_result_markdown(result_payload), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-correction-codex-draft.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "min_confidence": min_confidence, "write": True})
            manifest["transcript_semantic_correction_result_codex_json"] = "transcript-semantic-correction-result.codex.json"
            manifest["transcript_semantic_correction_result_codex_markdown"] = "transcript-semantic-correction-result.codex.md"
            manifest["transcript_semantic_correction_codex_draft_summary"] = {"status": status, "decision_count": len(decisions), "review_suggestion_count": len(review_suggestions), "updated_at": result_payload["updated_at"]}
            manifest["mcp_transcript_semantic_correction_codex_draft_args"] = "mcp-transcript-semantic-correction-codex-draft.args.json"
            write_json(root / "manifest.json", manifest)
    return result


def build_transcript_semantic_correction_llm_draft(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    execute: bool = False,
    limit: int = 80,
    min_confidence: float = 0.88,
    write: bool = True,
    business_authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build or execute an LLM review draft for semantic correction candidates.

    Default mode is preview-only. With execute=True, this calls the configured
    OpenAI-compatible text provider and stores a result JSON that still must pass
    validate_transcript_semantic_correction before closure can write anything.
    """
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = Path(input_json).expanduser().resolve() if input_json else root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    all_candidates = [row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    ordered_candidates, selection_summary = _prioritise_candidates_for_llm(all_candidates, root=root)
    candidates = ordered_candidates
    if limit and limit > 0:
        candidates = candidates[: int(limit)]
    selection_summary["selected_candidate_count"] = len(candidates)
    selection_summary["selected_candidate_ids"] = [str(row.get("candidate_id") or "") for row in candidates]
    prompt = _render_llm_draft_prompt(root=root, pack_path=pack_path, pack=pack if isinstance(pack, dict) else {}, candidates=candidates, min_confidence=min_confidence, selection_summary=selection_summary)
    gateway_pack = _build_transcript_semantic_gateway_pack(
        root=root,
        pack_path=pack_path,
        pack=pack if isinstance(pack, dict) else {},
        candidates=candidates,
        selection_summary=selection_summary,
    )
    gateway_pack_path = root / "transcript-semantic-correction-gateway-pack.json"
    gateway_pack_bytes = len(json.dumps(gateway_pack, ensure_ascii=False, indent=2).encode("utf-8"))
    source_pack_bytes = pack_path.stat().st_size if pack_path.is_file() else 0
    route_config = resolve_model_api_provider_config(
        "transcript_correction", provider_config
    )
    cfg = resolve_text_provider_config(route_config) if route_config else {}
    provider_public = _public_text_provider_config(cfg)
    result_payload: dict[str, Any] | None = None
    deep_validation: dict[str, Any] | None = None
    raw_model_output = ""
    error = ""
    status = "planned"
    if execute:
        if not candidates:
            result_payload = _normalise_llm_result_payload(
                {"decisions": [], "review_suggestions": []},
                pack_path=pack_path,
                min_confidence=min_confidence,
            )
            result_payload["candidate_selection"] = selection_summary
            deep_validation = validate_transcript_semantic_model_output(
                result_payload,
                pack if isinstance(pack, dict) else {},
                min_confidence=min_confidence,
                allow_empty_decisions=True,
            )
            status = "no_eligible_candidates"
        elif not cfg:
            status = "missing_provider_config"
            error = "execute=true requires a configured transcript_correction route"
        elif _uses_remote_proxy(cfg) and not business_authorization_path:
            status = "business_authorization_required"
            error = "remote proxy execution requires --business-authorization"
        elif _uses_remote_proxy(cfg) and not write:
            status = "business_authorization_write_required"
            error = "remote proxy execution requires write=true to create a child consent reservation"
        else:
            if _uses_remote_proxy(cfg):
                business = _prepare_semantic_business_context(
                    root, manifest, cfg, business_authorization_path, pack_path
                )
                if write:
                    write_json(gateway_pack_path, gateway_pack)
                child = create_business_child_consent(
                    business["authorization_path"],
                    stage_id=business["stage_id"],
                    artifact_paths=[gateway_pack_path],
                    producer="transcript_semantic_correction_gateway_pack",
                    input_paths=business["lineage_input_paths"],
                    max_calls=1,
                    write=True,
                )
                execution = execute_consented_model_task(
                    child["consent_path"],
                    expected_route_revision=business["route_revision"],
                    write=True,
                )
                model_result = execution.get("model_result") if isinstance(execution.get("model_result"), dict) else {}
                response = {
                    "ok": bool(execution.get("ok")),
                    "content": str(model_result.get("content") or ""),
                    "error": str(execution.get("error") or model_result.get("error") or ""),
                    "business_child_consent": _public_business_child_consent(child),
                    "connector_status": str(execution.get("status") or ""),
                }
            else:
                response = model_task_api_call(
                    "transcript_semantic_correction", provider_config=cfg,
                    messages=[
                        {"role": "system", "content": "You are a conservative transcript semantic correction reviewer. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    execute=True, temperature=0, response_format={"type": "json_object"},
                    max_tokens=4096, write=False,
                )
            raw_model_output = str(response.get("content") or "")
            if not response.get("ok"):
                status = str(response.get("error") or "provider_failed")
                error = str(response.get("error") or "provider_failed")
            else:
                try:
                    parsed = extract_json_document(raw_model_output, require_object=True)
                    result_payload = _normalise_llm_result_payload(parsed if isinstance(parsed, dict) else {}, pack_path=pack_path, min_confidence=min_confidence)
                    deep_validation = validate_transcript_semantic_model_output(
                        result_payload,
                        pack if isinstance(pack, dict) else {},
                        min_confidence=min_confidence,
                        allow_empty_decisions=not candidates,
                    )
                    if deep_validation["quality_gate_passed"]:
                        status = "executed"
                    else:
                        status = "model_output_contract_failed"
                        error = str(deep_validation.get("status") or status)
                except Exception as exc:
                    status = "model_output_parse_failed"
                    error = str(exc)
    if result_payload is None:
        result_payload = {
            "schema": RESULT_SCHEMA,
            "source": "text_llm_semantic_review" if execute else "text_llm_semantic_review_plan",
            "pack_json": str(pack_path),
            "min_confidence": float(min_confidence),
            "decisions": [],
            "review_suggestions": [],
            "operator_boundary": {
                "preview_by_default": True,
                "execute_may_call_text_llm": True,
                "does_not_modify_raw_sources": True,
                "validation_required_before_closure": True,
            },
            "updated_at": now_iso(),
        }
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_llm_draft.v1",
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"planned", "executed", "no_eligible_candidates"},
        "execute": bool(execute),
        "pack_json": str(pack_path),
        "candidate_count": len(candidates),
        "total_candidate_count": len(all_candidates),
        "candidate_selection": selection_summary,
        "decision_count": len(result_payload.get("decisions") or []),
        "review_suggestion_count": len(result_payload.get("review_suggestions") or []),
        "provider": provider_public,
        "prompt_markdown": str(root / "transcript-semantic-correction-llm-prompt.md"),
        "gateway_pack_json": str(gateway_pack_path),
        "gateway_pack_bytes": gateway_pack_bytes,
        "source_pack_bytes": source_pack_bytes,
        "input_byte_reduction_ratio": round(1 - gateway_pack_bytes / source_pack_bytes, 4) if source_pack_bytes else 0.0,
        "result_json": str(root / "transcript-semantic-correction-result.llm.json"),
        "result_markdown": str(root / "transcript-semantic-correction-result.llm.md"),
        "error": error,
        "deep_validation": deep_validation or {},
        "business_authorization": {
            "path": str(Path(business_authorization_path).expanduser().resolve()) if business_authorization_path else "",
            "required_for_remote_proxy": _uses_remote_proxy(cfg),
            "execution_mode": "business_child_consent" if business_authorization_path else "direct_or_local",
            "child_consent": response.get("business_child_consent", {}) if execute and "response" in locals() and isinstance(response, dict) else {},
            "connector_status": response.get("connector_status", "") if execute and "response" in locals() and isinstance(response, dict) else "",
        },
        "raw_model_output_saved": bool(execute and raw_model_output and status == "model_output_parse_failed"),
        "operator_boundary": result_payload.get("operator_boundary", {}),
        "updated_at": result_payload.get("updated_at") or now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_correction_llm_draft", timeout_seconds=1.0):
            (root / "transcript-semantic-correction-llm-prompt.md").write_text(prompt, encoding="utf-8")
            write_json(gateway_pack_path, gateway_pack)
            write_json(root / "mcp-transcript-semantic-correction-llm-draft.args.json", {"bundle_dir": str(root), "input_json": str(pack_path), "provider_config": {}, "execute": False, "limit": int(limit or 0), "min_confidence": min_confidence, "write": True})
            if execute and raw_model_output and status == "model_output_parse_failed":
                (root / "transcript-semantic-correction-result.llm.raw.txt").write_text(raw_model_output, encoding="utf-8")
                manifest["transcript_semantic_correction_result_llm_raw_text"] = "transcript-semantic-correction-result.llm.raw.txt"
            if execute and status in {"executed", "no_eligible_candidates"}:
                write_json(root / "transcript-semantic-correction-result.llm.json", result_payload)
                (root / "transcript-semantic-correction-result.llm.md").write_text(_render_llm_result_markdown(result_payload), encoding="utf-8")
                manifest["transcript_semantic_correction_result_llm_json"] = "transcript-semantic-correction-result.llm.json"
                manifest["transcript_semantic_correction_result_llm_markdown"] = "transcript-semantic-correction-result.llm.md"
            manifest["transcript_semantic_correction_llm_prompt_markdown"] = "transcript-semantic-correction-llm-prompt.md"
            manifest["transcript_semantic_correction_gateway_pack_json"] = "transcript-semantic-correction-gateway-pack.json"
            manifest["transcript_semantic_correction_llm_draft_summary"] = {"status": status, "candidate_count": len(candidates), "total_candidate_count": len(all_candidates), "candidate_selection": selection_summary, "decision_count": result["decision_count"], "review_suggestion_count": result["review_suggestion_count"], "updated_at": result["updated_at"]}
            manifest["mcp_transcript_semantic_correction_llm_draft_args"] = "mcp-transcript-semantic-correction-llm-draft.args.json"
            write_json(root / "manifest.json", manifest)
    return result


def _build_transcript_semantic_gateway_pack(
    *,
    root: Path,
    pack_path: Path,
    pack: dict[str, Any],
    candidates: list[dict[str, Any]],
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    """Create the consent-safe compact pack consumed by the model gateway."""
    return {
        "schema": PACK_SCHEMA,
        "pack_profile": "gateway_compact_v1",
        "bundle_dir": str(root),
        "title": pack.get("title") or root.name,
        "source_pack_json": str(pack_path),
        "source_pack_sha256": _sha256_file(pack_path),
        "source_candidate_count": len(pack.get("candidates") or []),
        "candidate_selection": selection_summary,
        "candidates": [_compact_candidate_for_llm(row) for row in candidates],
        "operator_boundary": {
            "candidate_subset_only": True,
            "does_not_replace_source_pack": True,
            "must_validate_against_source_pack_before_closure": True,
        },
        "updated_at": now_iso(),
    }


def _normalise_llm_result_payload(payload: dict[str, Any], *, pack_path: Path, min_confidence: float) -> dict[str, Any]:
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    review_suggestions = payload.get("review_suggestions") if isinstance(payload.get("review_suggestions"), list) else []
    return {
        "schema": RESULT_SCHEMA,
        "source": str(payload.get("source") or "text_llm_semantic_review"),
        "pack_json": str(pack_path),
        "min_confidence": float(payload.get("min_confidence") or min_confidence),
        "decisions": [row for row in decisions if isinstance(row, dict)],
        "review_suggestions": [row for row in review_suggestions if isinstance(row, dict)],
        "operator_boundary": {
            "execute_may_call_text_llm": True,
            "does_not_modify_raw_sources": True,
            "validation_required_before_closure": True,
            "api_key_not_persisted": True,
        },
        "updated_at": now_iso(),
    }


def _public_text_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "base_url": cfg.get("base_url", ""),
        "api_key_configured": bool(cfg.get("api_key")),
        "interface": cfg.get("interface", "openai_chat_completions"),
    }


def _uses_remote_proxy(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("execution_location") or "").strip().lower() == "remote" and str(cfg.get("adapter_backend") or "").strip().lower() == "proxy"


def _prepare_semantic_business_context(root: Path, manifest: dict[str, Any], cfg: dict[str, Any], authorization_path: str | Path | None, pack_path: Path) -> dict[str, Any]:
    if authorization_path is None:
        raise ValueError("remote proxy execution requires a business authorization")
    path = Path(authorization_path).expanduser().resolve()
    status = validate_model_business_authorization(path)
    if not status.get("valid"):
        blockers = [str(row.get("key") or "blocked") for row in status.get("blockers") or [] if isinstance(row, dict)]
        raise ValueError("business authorization is not active: " + (",".join(blockers) or "unknown"))
    bundle_dirs = {Path(str(value or "")).expanduser().resolve() for value in (status.get("bundle_dirs") or [status.get("bundle_dir")]) if str(value or "").strip()}
    if root not in bundle_dirs:
        raise ValueError("business authorization bundle does not match bundle_dir")
    payload = read_json(path)
    route_id = str(cfg.get("route_id") or "")
    route_revision = str(cfg.get("route_revision") or "")
    matches = []
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict) or str(stage.get("task") or "") != "transcript_semantic_correction":
            continue
        route = stage.get("route_snapshot") if isinstance(stage.get("route_snapshot"), dict) else {}
        if route_id and str(route.get("route_id") or "") != route_id:
            continue
        if route_revision and str(route.get("route_revision") or "") != route_revision:
            continue
        matches.append(stage)
    if len(matches) != 1:
        raise ValueError("business authorization must contain exactly one matching transcript_semantic_correction route stage")
    stage = matches[0]
    if "transcript_semantic_correction_gateway_pack" not in [str(value) for value in stage.get("allowed_producers") or []]:
        raise ValueError("business authorization semantic stage does not allow transcript_semantic_correction_gateway_pack")
    transcript = _semantic_canonical_transcript_path(root, manifest)
    known_paths = {str(row.get("path") or "") for row in payload.get("sources") or [] if isinstance(row, dict)}
    if str(transcript) not in known_paths or str(pack_path) not in known_paths:
        raise ValueError("business authorization does not bind the canonical transcript and semantic correction pack")
    route = stage.get("route_snapshot") if isinstance(stage.get("route_snapshot"), dict) else {}
    return {"authorization_path": str(path), "stage_id": str(stage.get("id") or ""), "route_revision": str(route.get("route_revision") or ""), "lineage_input_paths": [str(transcript), str(pack_path)]}


def _semantic_canonical_transcript_path(root: Path, manifest: dict[str, Any]) -> Path:
    for key in ("source_arbitrated_transcript_json", "corrected_transcript_json", "normalized_transcript_json", "transcript_json"):
        value = str(manifest.get(key) or "").strip()
        path = _bundle_path(root, value) if value else Path()
        if path.is_file():
            return path.resolve()
    fallback = root / "normalized-transcript.json"
    if fallback.is_file():
        return fallback.resolve()
    raise ValueError("canonical transcript is unavailable for semantic correction")


def _public_business_child_consent(child: dict[str, Any]) -> dict[str, str]:
    return {key: str(child.get(key) or "") for key in ("status", "consent_path", "consent_id", "route_revision", "admission_id")}


def _render_llm_draft_prompt(*, root: Path, pack_path: Path, pack: dict[str, Any], candidates: list[dict[str, Any]], min_confidence: float, selection_summary: dict[str, Any] | None = None) -> str:
    payload = {
        "bundle_dir": str(root),
        "pack_json": str(pack_path),
        "title": pack.get("title") or root.name,
        "min_confidence": float(min_confidence),
        "candidate_selection": selection_summary or {},
        "output_schema": _result_template({"candidates": candidates}),
        "output_contract": transcript_semantic_correction_output_contract(),
        "rules": [
            "Only judge listed candidates; do not freely rewrite the whole transcript.",
            "Use ASR/subtitle, OCR/ebook, visual, timeline, tagger, and metadata evidence together.",
            "Use action=replace only when the semantic correction is strongly supported by evidence.",
            "Numbers, prices, names, dates, percentages, and claims are high risk; mark needs_human_review unless evidence is decisive.",
            "For safe global acronym/proper-noun normalization, set apply_scope=all_segments.",
            "If uncertain, use action=needs_human_review or action=reject.",
            transcript_semantic_correction_model_instructions(),
        ],
        "candidates": [_compact_candidate_for_llm(row) for row in candidates],
    }
    return "# 转写语义纠错 LLM 判读任务\n\n请只返回 JSON，不要输出解释性正文。\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n"


def _prioritise_candidates_for_llm(candidates: list[dict[str, Any]], *, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chapters = _load_semantic_chapter_ranges(root)
    eligible_candidates = [row for row in candidates if _candidate_llm_review_eligible(row)]
    deferred_candidates = [row for row in candidates if not _candidate_llm_review_eligible(row)]
    attention = _semantic_attention_items(eligible_candidates, chapters)
    by_id = {str(row.get("candidate_id") or ""): row for row in eligible_candidates if str(row.get("candidate_id") or "")}
    ordered: list[dict[str, Any]] = []
    attention_ids: list[str] = []
    for row in attention:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = by_id.get(candidate_id)
        if not candidate or candidate_id in attention_ids:
            continue
        candidate = dict(candidate)
        candidate["llm_priority_score"] = int(row.get("priority_score") or 0)
        candidate["llm_priority_reason"] = str(row.get("reason") or candidate.get("reason") or "")
        ordered.append(candidate)
        attention_ids.append(candidate_id)
    selected = set(attention_ids)
    for candidate in eligible_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in selected:
            continue
        ordered.append(candidate)
    return ordered, {
        "strategy": "source_conflict_first",
        "total_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "deferred_low_evidence_candidate_count": len(deferred_candidates),
        "deferred_low_evidence_reason_counts": _count_values(row.get("llm_review_defer_reason") or row.get("reason") for row in deferred_candidates),
        "attention_candidate_count": len(attention_ids),
        "attention_candidate_ids": attention_ids,
    }


def _compact_candidate_for_llm(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_group_id": candidate.get("candidate_group_id"),
        "canonical_hint": candidate.get("canonical_hint"),
        "segment_index": candidate.get("segment_index"),
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "time_range": candidate.get("time_range"),
        "timeline_indexes": candidate.get("timeline_indexes") or [],
        "correction_type": candidate.get("correction_type"),
        "risk_level": candidate.get("risk_level"),
        "llm_priority_score": candidate.get("llm_priority_score"),
        "llm_priority_reason": candidate.get("llm_priority_reason"),
        "original_text": candidate.get("original_text"),
        "candidate_text": candidate.get("candidate_text"),
        "suggested_text": candidate.get("suggested_text"),
        "context_text": str(candidate.get("context_text") or "")[:800],
        "context_window": candidate.get("context_window") or {},
        "speaker": candidate.get("speaker") or "",
        "speaker_label": candidate.get("speaker_label") or "",
        "speaker_role": candidate.get("speaker_role") or "",
        "chapter_index": candidate.get("chapter_index"),
        "chapter_title": candidate.get("chapter_title") or "",
        "chapter_time_range": candidate.get("chapter_time_range") or "",
        "edit_contract": candidate.get("edit_contract") or {},
        "reason": candidate.get("reason"),
        "has_conflict": candidate.get("has_conflict"),
        "needs_human_review": candidate.get("needs_human_review"),
        "evidence_ids": candidate.get("evidence_ids") or [],
        "evidence": [
            {"evidence_id": item.get("evidence_id"), "source_type": item.get("source_type"), "text": str(item.get("text") or "")[:500]}
            for item in (candidate.get("evidence") or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _render_llm_result_markdown(payload: dict[str, Any]) -> str:
    return "\n".join([
        "# transcript-semantic-correction-result.llm.md",
        "",
        "本文件由 text LLM provider 生成，仍必须经过 validate-transcript-semantic-correction 校验后才能进入 closure。",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ])


def validate_transcript_semantic_model_output(
    content: Any,
    pack: dict[str, Any],
    *,
    min_confidence: float = 0.88,
    allow_empty_decisions: bool = False,
) -> dict[str, Any]:
    """Deep, non-persisting validation used by the Trusted Connector."""
    contract_issues: list[dict[str, str]] = []
    if not isinstance(pack, dict) or pack.get("schema") != PACK_SCHEMA:
        contract_issues.append(
            {"key": "semantic_pack_invalid", "detail": f"expected {PACK_SCHEMA}"}
        )
    try:
        payload = (
            content
            if isinstance(content, dict)
            else extract_json_document(str(content or ""), require_object=True)
        )
    except Exception as exc:
        payload = {}
        contract_issues.append(
            {"key": "semantic_result_parse_failed", "detail": str(exc)}
        )
    if not isinstance(payload, dict):
        payload = {}
        contract_issues.append(
            {"key": "semantic_result_not_object", "detail": "result must be an object"}
        )
    if payload.get("schema") != RESULT_SCHEMA:
        contract_issues.append(
            {"key": "semantic_result_schema_mismatch", "detail": f"expected {RESULT_SCHEMA}"}
        )
    if not str(payload.get("source") or "").strip():
        contract_issues.append(
            {"key": "semantic_result_source_missing", "detail": "source"}
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        decisions = []
        contract_issues.append(
            {"key": "semantic_result_decisions_invalid", "detail": "decisions must be an array"}
        )
    for index, decision in enumerate(decisions):
        contract_issues.extend(_strict_decision_contract_issues(decision, index=index))

    candidates = {
        str(row.get("candidate_id")): row
        for row in (pack.get("candidates") if isinstance(pack, dict) else []) or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        row = _validate_decision(
            decision,
            candidates,
            idx=index,
            min_confidence=min_confidence,
            strict_model_contract=True,
        )
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id and candidate_id in seen_ids:
            row = dict(row)
            row["accepted"] = False
            row["arbitrated_no_change"] = False
            row["semantic_decision_status"] = "rejected"
            row["reject_reasons"] = [
                *list(row.get("reject_reasons") or []),
                "duplicate_candidate_id",
            ]
        if candidate_id:
            seen_ids.add(candidate_id)
        rows.append(row)
    accepted = [row for row in rows if row.get("accepted")]
    no_change = [row for row in rows if row.get("arbitrated_no_change")]
    rejected = [
        row
        for row in rows
        if not row.get("accepted") and not row.get("arbitrated_no_change")
    ]
    quality_issues = [
        {"key": str(reason), "detail": f"decisions[{index}]"}
        for index, row in enumerate(rows)
        for reason in row.get("reject_reasons") or []
    ]
    if not rows and not allow_empty_decisions:
        quality_issues.append(
            {"key": "semantic_decisions_empty", "detail": "at least one decision is required"}
        )
    contract_ok = not contract_issues
    quality_gate_passed = contract_ok and (bool(rows) or allow_empty_decisions) and not rejected
    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_model_output_validation.v1",
        "status": (
            "qualified"
            if quality_gate_passed
            else ("contract_failed" if not contract_ok else "quality_gate_failed")
        ),
        "contract_ok": contract_ok,
        "quality_gate_passed": quality_gate_passed,
        "decision_count": len(rows),
        "accepted_decision_count": len(accepted),
        "arbitrated_no_change_count": len(no_change),
        "rejected_decision_count": len(rejected),
        "contract_issues": contract_issues,
        "quality_issues": quality_issues,
        "content_persisted": False,
    }


def _strict_decision_contract_issues(
    decision: Any, *, index: int
) -> list[dict[str, str]]:
    prefix = f"decisions[{index}]"
    if not isinstance(decision, dict):
        return [{"key": "semantic_decision_not_object", "detail": prefix}]
    issues: list[dict[str, str]] = []
    for key, expected_type in STRICT_DECISION_REQUIRED_KEYS.items():
        detail = f"{prefix}.{key}"
        if key not in decision:
            issues.append({"key": "semantic_decision_field_missing", "detail": detail})
            continue
        value = decision[key]
        type_ok = {
            "string": isinstance(value, str),
            "array": isinstance(value, list),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }[expected_type]
        if not type_ok:
            issues.append(
                {
                    "key": "semantic_decision_field_type_mismatch",
                    "detail": f"{detail}: expected {expected_type}",
                }
            )
    return issues


def validate_transcript_semantic_correction(bundle_dir: str | Path, *, input_json: str | Path, min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    candidates = {str(row.get("candidate_id")): row for row in (pack.get("candidates") if isinstance(pack, dict) else []) if isinstance(row, dict)}
    imported = _load_import(input_json)
    rows = []
    for idx, decision in enumerate(imported.get("decisions") or imported.get("corrections") or []):
        rows.append(_validate_decision(decision, candidates, idx=idx, min_confidence=min_confidence))
    accepted = [row for row in rows if row.get("accepted")]
    no_change = [row for row in rows if row.get("arbitrated_no_change")]
    rejected = [row for row in rows if not row.get("accepted") and not row.get("arbitrated_no_change")]
    review_rows = _semantic_review_rows(rejected)
    selection = (
        imported.get("candidate_selection")
        if isinstance(imported.get("candidate_selection"), dict)
        else {}
    )
    no_eligible_candidates = not rows and str(selection.get("selected_candidate_count")) == "0"
    if accepted and not rejected:
        status = "accepted_with_no_change" if no_change else "accepted"
    elif accepted:
        status = "accepted_with_rejections"
    elif no_change and not rejected:
        status = "arbitrated_no_change"
    elif no_eligible_candidates:
        status = "no_eligible_candidates"
    else:
        status = "rejected"
    result = {
        "schema": VALIDATION_SCHEMA,
        "bundle_dir": str(root),
        "input_json": str(Path(input_json).expanduser().resolve()),
        "input_sha256": _sha256_file(Path(input_json).expanduser().resolve()),
        "status": status,
        "ok": status in {"accepted", "accepted_with_no_change", "accepted_with_rejections", "arbitrated_no_change", "no_eligible_candidates"},
        "min_confidence": float(min_confidence),
        "pack_path": str(pack_path),
        "pack_sha256": _sha256_file(pack_path),
        "pack_candidate_count": len(candidates),
        "decision_count": len(rows),
        "accepted_decision_count": len(accepted),
        "arbitrated_no_change_count": len(no_change),
        "rejected_decision_count": len(rejected),
        "review_required_count": len(review_rows),
        "review_rows": review_rows,
        "accepted_decisions": accepted,
        "arbitrated_no_change_decisions": no_change,
        "rejected_decisions": rejected,
        "decisions": rows,
        "artifacts": {"json": str(root / "transcript-semantic-correction-validation.json"), "markdown": str(root / "transcript-semantic-correction-validation.md")},
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "does_not_modify_raw_sources": True, "numbers_require_stronger_evidence": True},
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="validate_transcript_semantic_correction", timeout_seconds=1.0):
            validation_path = root / "transcript-semantic-correction-validation.json"
            previous = _read_optional_json(validation_path)
            # Reuse the exact receipt when only its observation time changed.
            # Canonical transcripts bind this file's SHA-256: regenerating an
            # identical validation every second falsely invalidates exports.
            # Pack/input hashes, thresholds and every decision remain compared;
            # never suppress an actual evidence or authorization change.
            if _canonical_payload_equivalent(previous, result):
                result = previous
            else:
                write_json(validation_path, result)
            (root / "transcript-semantic-correction-validation.md").write_text(_render_validation_markdown(result), encoding="utf-8")
            write_json(root / "transcript-semantic-correction-review.json", _semantic_review_payload(root, review_rows))
            (root / "transcript-semantic-correction-review.md").write_text(_render_semantic_review_markdown(review_rows), encoding="utf-8")
            manifest["transcript_semantic_correction_validation_json"] = "transcript-semantic-correction-validation.json"
            manifest["transcript_semantic_correction_validation_markdown"] = "transcript-semantic-correction-validation.md"
            manifest["transcript_semantic_correction_review_json"] = "transcript-semantic-correction-review.json"
            manifest["transcript_semantic_correction_review_markdown"] = "transcript-semantic-correction-review.md"
            manifest["transcript_semantic_correction_validation_summary"] = {"status": status, "accepted_decision_count": len(accepted), "arbitrated_no_change_count": len(no_change), "rejected_decision_count": len(rejected), "review_required_count": len(review_rows), "updated_at": result["updated_at"]}
            write_json(root / "manifest.json", manifest)
    return result


def import_transcript_semantic_review_notes(
    bundle_dir: str | Path,
    *,
    review_json: str | Path | None = None,
    min_confidence: float = 0.88,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack = _read_optional_json(pack_path)
    candidates = {str(row.get("candidate_id")): row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)}
    review_path = Path(review_json).expanduser().resolve() if review_json else root / "transcript-semantic-correction-review-notes.json"
    if not review_path.exists() and review_json is None:
        review_path = root / "review-notes.json"
    if not review_path.exists():
        raise FileNotFoundError(f"semantic correction review notes not found: {review_path}")
    payload = _load_import(review_path)
    rows = _review_note_rows(payload)
    decisions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        decision = _decision_from_review_row(row, candidates)
        if decision is None:
            skipped.append({"row_number": idx, "candidate_id": row.get("candidate_id"), "reason": "missing_or_unknown_candidate_or_no_review_decision"})
            continue
        decisions.append(decision)
    result_payload = {
        "schema": RESULT_SCHEMA,
        "source": "human_review_notes",
        "review_json": str(review_path),
        "import_summary": {
        "pack_identity": {
            "path": str(pack_path),
            "sha256": _sha256_file(pack_path),
            "candidate_count": len(candidates),
        },
            "status": "imported" if decisions else "no_importable_decisions",
            "decision_count": len(decisions),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "updated_at": now_iso(),
        },
        "decisions": decisions,
    }
    output_json = root / "transcript-semantic-correction-result.review.json"
    output_md = root / "transcript-semantic-correction-result.review.md"
    validation: dict[str, Any] = {}
    if write:
        with bundle_write_lock(root, operation="import_transcript_semantic_review_notes", timeout_seconds=1.0):
            write_json(output_json, result_payload)
            output_md.write_text(_render_review_import_markdown(result_payload, skipped), encoding="utf-8")
            manifest["transcript_semantic_correction_result_review_json"] = output_json.name
            manifest["transcript_semantic_correction_result_review_markdown"] = output_md.name
            manifest["mcp_import_transcript_semantic_review_notes_args"] = "mcp-import-transcript-semantic-review-notes.args.json"
            write_json(root / "mcp-import-transcript-semantic-review-notes.args.json", {"bundle_dir": str(root), "review_json": str(review_path), "min_confidence": min_confidence, "write": True})
            write_json(root / "manifest.json", manifest)
        validation = validate_transcript_semantic_correction(root, input_json=output_json, min_confidence=min_confidence, write=True)
    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_review_import.v1",
        "bundle_dir": str(root),
        "review_json": str(review_path),
        "status": "imported" if decisions else "no_importable_decisions",
        "ok": bool(decisions),
        "decision_count": len(decisions),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "result_json": str(output_json),
        "result_markdown": str(output_md),
        "validation": validation,
        "operator_boundary": {"local_only": True, "no_cloud_call": True, "does_not_modify_raw_sources": True, "closure_still_required": True},
        "updated_at": now_iso(),
    }

def transcript_semantic_correction_closure(
    bundle_dir: str | Path,
    *,
    input_json: str | Path,
    min_confidence: float = 0.88,
    auto_apply: bool = False,
    refresh_exports: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    validation = validate_transcript_semantic_correction(root, input_json=input_json, min_confidence=min_confidence, write=write)
    current_accepted = [row for row in validation.get("accepted_decisions", []) if isinstance(row, dict)]
    if not current_accepted:
        no_change_count = int(validation.get("arbitrated_no_change_count") or 0)
        rejected_count = int(validation.get("rejected_decision_count") or 0)
        no_eligible = str(validation.get("status") or "") == "no_eligible_candidates"
        status = (
            "completed_no_text_changes"
            if rejected_count == 0 and (no_change_count > 0 or no_eligible)
            else "no_safe_decisions"
        )
        corrected_payload: dict[str, Any] | None = None
        passthrough_segments: list[dict[str, Any]] = []
        if status == "completed_no_text_changes":
            timeline = _read_optional_json(root / "timeline.json")
            cues = _load_semantic_closure_cues(
                root, manifest, timeline if isinstance(timeline, list) else []
            )
            passthrough_segments, _ = _apply_decisions_to_cues(cues, [])
            corrected_payload = {
                "schema": CORRECTED_SCHEMA,
                "bundle_dir": str(root),
                "source": "transcript_semantic_correction_no_change",
                "updated_at": now_iso(),
                "summary": {
                    "segments": len(passthrough_segments),
                    "changed_segments": 0,
                    "applied_corrections": 0,
                    "promoted_to_corrected_transcript": True,
                },
                "semantic_correction": {
                    "validation_json": "transcript-semantic-correction-validation.json",
                    "min_confidence": float(min_confidence),
                    "auto_apply": bool(auto_apply),
                    "current_accepted_decision_count": 0,
                    "cumulative_decision_count": 0,
                    "decisions": [],
                    "closed_without_text_change": [],
                    "reason": "no_eligible_candidates" if no_eligible else "arbitrated_no_change",
                },
                "segments": passthrough_segments,
            }
        result = _closure_result(root, manifest, validation, passthrough_segments, [], status=status, auto_apply=auto_apply)
        result["refresh_exports_requested"] = bool(refresh_exports)
        result["refresh_exports_status"] = "skipped_no_applied_corrections" if status == "completed_no_text_changes" else "skipped_no_safe_decisions"
        if write:
            result["transaction_receipt"] = _write_closure(
                root,
                manifest,
                result,
                corrected_payload=corrected_payload,
            )
        return result
    decision_ledger = _merge_correction_decision_ledger(root, current_accepted, validation=validation)
    accepted = [row for row in decision_ledger.get("decisions", []) if isinstance(row, dict)]
    timeline = _read_optional_json(root / "timeline.json")
    cues = _load_semantic_closure_cues(root, manifest, timeline if isinstance(timeline, list) else [])
    corrected_segments, applied = _apply_decisions_to_cues(cues, accepted)
    replace_decision_count = sum(1 for row in accepted if row.get("action") == "replace")
    status = "completed" if applied else ("completed_no_text_changes" if accepted and replace_decision_count == 0 else "no_matching_segments")
    corrected_payload = {
        "schema": CORRECTED_SCHEMA,
        "bundle_dir": str(root),
        "source": "transcript_semantic_correction",
        "updated_at": now_iso(),
        "summary": {"segments": len(corrected_segments), "changed_segments": sum(1 for row in corrected_segments if row.get("changed")), "applied_corrections": len(applied), "promoted_to_corrected_transcript": True},
        "semantic_correction": {
            "validation_json": "transcript-semantic-correction-validation.json",
            "decision_ledger_json": "transcript-semantic-correction-decision-ledger.json",
            "min_confidence": float(min_confidence),
            "auto_apply": bool(auto_apply),
            "current_accepted_decision_count": len(current_accepted),
            "cumulative_decision_count": len(accepted),
            "decisions": applied,
            "closed_without_text_change": [row for row in accepted if row.get("action") == "keep_original"],
        },
        "segments": corrected_segments,
    }
    result = _closure_result(root, manifest, validation, corrected_segments, applied, status=status, auto_apply=auto_apply)
    result["current_accepted_decision_count"] = len(current_accepted)
    result["cumulative_decision_count"] = len(accepted)
    result["decision_ledger_json"] = str(root / "transcript-semantic-correction-decision-ledger.json")
    result["refresh_exports_requested"] = bool(refresh_exports)
    result["refresh_exports_status"] = (
        "pending_downstream_refresh"
        if refresh_exports and applied
        else ("skipped_no_applied_corrections" if refresh_exports else "not_requested")
    )
    if write:
        result["transaction_receipt"] = _write_closure(
            root,
            manifest,
            result,
            corrected_payload=corrected_payload if applied else None,
            decision_ledger=decision_ledger,
        )
        if refresh_exports and applied:
            refresh_result = _refresh_semantic_correction_outputs(
                root,
                canonical_before_sha256=str(result.get("canonical_before_sha256") or ""),
                canonical_after_sha256=str(result.get("canonical_after_sha256") or ""),
            )
            result["refresh_exports_status"] = refresh_result.get("status")
            result["refresh_exports"] = refresh_result
    return result

def _refresh_semantic_correction_outputs(
    root: Path,
    *,
    canonical_before_sha256: str,
    canonical_after_sha256: str,
) -> dict[str, Any]:
    from .transcript_downstream_refresh import refresh_transcript_downstream_outputs
    from .transcript_semantic_batch import transcript_semantic_acceptance
    from .transcript_semantic_summary_impact import transcript_semantic_summary_impact_report

    downstream = refresh_transcript_downstream_outputs(
        root,
        canonical_before_sha256=canonical_before_sha256,
        canonical_after_sha256=canonical_after_sha256,
        reason="transcript_semantic_correction_committed",
        write=True,
    )
    impact = transcript_semantic_correction_impact_report(root, write=True)
    readable_impact = transcript_semantic_correction_readable_impact_report(root, write=True)
    summary_impact = transcript_semantic_summary_impact_report(root, write=True)
    requires_summary_regeneration = bool(downstream.get("requires_summary_regeneration"))
    # The historical acceptance projection does not yet own the transcript-
    # triggered summary invalidation gate. Run it read-only while a summary is
    # stale, and expose the stronger composite state instead of persisting a
    # misleading ``accepted`` receipt.
    acceptance = transcript_semantic_acceptance(
        root,
        write=not requires_summary_regeneration,
    )
    reported_acceptance_status = (
        "pending_summary_regeneration"
        if requires_summary_regeneration
        else str(acceptance.get("status") or "needs_semantic_correction_action")
    )
    production_qualified = bool(
        downstream.get("full_pipeline_production_qualified")
        and impact.get("status") == "passed"
        and readable_impact.get("status") == "passed"
        and summary_impact.get("status") == "passed"
        and reported_acceptance_status == "accepted"
    )
    if production_qualified:
        status = "completed"
    elif requires_summary_regeneration:
        status = "needs_summary_regeneration"
    elif str(downstream.get("status") or "") == "degraded":
        status = "degraded"
    else:
        status = "needs_semantic_acceptance"
    return {
        "status": status,
        "ok": production_qualified,
        "downstream_refresh_status": downstream.get("status"),
        "downstream_refresh": downstream,
        "requires_summary_regeneration": requires_summary_regeneration,
        "full_pipeline_production_qualified": production_qualified,
        "impact_status": impact.get("status"),
        "readable_impact_status": readable_impact.get("status"),
        "summary_impact_status": summary_impact.get("status"),
        "semantic_acceptance_status": reported_acceptance_status,
        "semantic_acceptance_raw_status": acceptance.get("status"),
        "semantic_acceptance_written": not requires_summary_regeneration,
        "semantic_acceptance_path": acceptance.get("json_path"),
        "canonical_transcript_integrity": acceptance.get("canonical_transcript_integrity") or {},
        "readable_patch_status": "removed",
        "readable_patch_replacement_count": 0,
        "smart_summary_handling": "archived_and_invalidated_pending_consented_regeneration",
        "updated_at": now_iso(),
    }
def transcript_semantic_correction_impact_report(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    accepted = _accepted_correction_decisions(root)
    docs = _load_output_documents(root, manifest)
    correction_rows = []
    final_residual_total = 0
    corrected_hit_total = 0
    final_keys = {"source_arbitrated_transcript_json", "source_arbitrated_transcript_markdown", "full_transcript", "full_body", "smart_summary", "smart_summary_codex", "content_candidate_pack", "content_material_card"}
    for row in accepted:
        original = str(row.get("original_text") or "").strip()
        corrected = str(row.get("corrected_text") or "").strip()
        if not original or not corrected or original == corrected:
            continue
        by_doc = {}
        for doc in docs:
            text = _impact_countable_text(str(doc.get("text") or ""), str(doc.get("key") or ""))
            by_doc[doc["key"]] = {"original_count": _count_text(text, original), "corrected_count": _count_text(text, corrected), "path": doc.get("path")}
        residual = sum(v["original_count"] for key, v in by_doc.items() if key in final_keys)
        hits = sum(v["corrected_count"] for key, v in by_doc.items() if key in final_keys)
        final_residual_total += residual
        corrected_hit_total += hits
        correction_rows.append({"candidate_id": row.get("candidate_id"), "correction_type": row.get("correction_type"), "original_text": original, "corrected_text": corrected, "final_residual_count": residual, "final_corrected_count": hits, "by_document": by_doc})
    status = "no_accepted_decisions" if not accepted else ("passed" if final_residual_total == 0 else "needs_fix")
    result = {
        "schema": IMPACT_SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": status in {"passed", "no_accepted_decisions"},
        "accepted_decision_count": len(accepted),
        "final_residual_error_total": final_residual_total,
        "final_corrected_hit_total": corrected_hit_total,
        "documents": [{k: v for k, v in doc.items() if k != "text"} for doc in docs],
        "corrections": correction_rows,
        "artifacts": {"json": str(root / "transcript-semantic-correction-impact-report.json"), "markdown": str(root / "transcript-semantic-correction-impact-report.md")},
        "next_actions": _impact_next_actions(status),
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_correction_impact_report", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-correction-impact-report.json", result)
            (root / "transcript-semantic-correction-impact-report.md").write_text(_render_impact_markdown(result), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-correction-impact-report.args.json", {"bundle_dir": str(root), "write": True})
            manifest["transcript_semantic_correction_impact_report_json"] = "transcript-semantic-correction-impact-report.json"
            manifest["transcript_semantic_correction_impact_report_markdown"] = "transcript-semantic-correction-impact-report.md"
            manifest["mcp_transcript_semantic_correction_impact_report_args"] = "mcp-transcript-semantic-correction-impact-report.args.json"
            manifest["transcript_semantic_correction_impact_summary"] = {"status": status, "accepted_decision_count": len(accepted), "final_residual_error_total": final_residual_total, "updated_at": result["updated_at"]}
            write_json(root / "manifest.json", manifest)
    return result



def _is_final_llm_summary_text(text: str) -> bool:
    final_modes = (
        "codex_final",
        "codex_llm_rewrite_final",
        "codex_llm_rewrite_substitute",
        "codex_first_llm_substitute",
        "online_llm_section_rewrite",
    )
    pattern = r"(?m)^\s*生成方式\s*[：:]\s*\x60?(?:" + "|".join(re.escape(mode) for mode in final_modes) + r")\b"
    return bool(re.search(pattern, text))

def transcript_semantic_correction_readable_impact_report(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Report whether semantic corrections reached human-readable exports."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    accepted = _accepted_correction_decisions(root)
    docs = _load_readable_output_documents(root, manifest)
    rows = []
    smart_summary_doc = next((row for row in docs if row.get("key") == "smart_summary"), None)
    smart_summary_is_final = _is_final_llm_summary_text(str((smart_summary_doc or {}).get("text") or ""))
    required_keys = {"full_transcript"}
    if smart_summary_is_final:
        required_keys.add("smart_summary")
    readable_keys = {
        str(doc.get("key")) for doc in docs
        if _impact_countable_text(str(doc.get("text") or ""), str(doc.get("key") or "")).strip()
    }
    missing_required_documents = sorted(required_keys - readable_keys)
    required_residual_total = 0
    for row in accepted:
        original = str(row.get("original_text") or "").strip()
        corrected = str(row.get("corrected_text") or "").strip()
        if not original or not corrected or original == corrected:
            continue
        by_doc = {}
        for doc in docs:
            clean = _impact_countable_text(str(doc.get("text") or ""), str(doc.get("key") or ""))
            original_count = _count_text(clean, original)
            corrected_count = _count_text(clean, corrected)
            by_doc[doc["key"]] = {
                "original_count": original_count,
                "corrected_count": corrected_count,
                "path": doc.get("path"),
                "role": doc.get("role"),
                "sample_corrected_lines": _sample_lines(clean, corrected, limit=3),
                "sample_residual_lines": _sample_lines(clean, original, limit=3),
            }
        residual = sum(v["original_count"] for key, v in by_doc.items() if key in required_keys)
        required_residual_total += residual
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "original_text": original,
            "corrected_text": corrected,
            "correction_type": row.get("correction_type"),
            "required_readable_residual_count": residual,
            "by_document": by_doc,
        })
    if not accepted:
        status = "no_accepted_decisions"
    elif missing_required_documents:
        status = "missing_required_documents"
    else:
        status = "passed" if required_residual_total == 0 else "needs_fix"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_readable_impact.v1",
        "bundle_dir": str(root),
        "status": status,
        "ok": status == "passed",
        "accepted_decision_count": len(accepted),
        "required_documents": sorted(required_keys),
        "missing_required_documents": missing_required_documents,
        "smart_summary_evaluation": "required_final_llm_summary" if smart_summary_is_final else "pending_llm_not_required",
        "required_readable_residual_total": required_residual_total,
        "corrections": rows,
        "documents": [{"key": row.get("key"), "path": row.get("path"), "role": row.get("role")} for row in docs],
        "notes": [
            "Missing or empty required readable documents cannot establish successful correction propagation.",
            "full_transcript must absorb high-confidence corrections.",
            "smart_summary is required only after an LLM-generated final summary exists; a needs_llm_summary placeholder is not evaluated.",
            "knowledge_note is reported but not a failing document because it may intentionally preserve raw evidence/audit text.",
        ],
        "artifacts": {"json": str(root / "transcript-semantic-readable-impact-report.json"), "markdown": str(root / "transcript-semantic-readable-impact-report.md")},
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_readable_impact_report", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-readable-impact-report.json", result)
            (root / "transcript-semantic-readable-impact-report.md").write_text(_render_readable_impact_markdown(result), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-readable-impact-report.args.json", {"bundle_dir": str(root), "write": True})
            manifest["transcript_semantic_correction_readable_impact_json"] = "transcript-semantic-readable-impact-report.json"
            manifest["transcript_semantic_correction_readable_impact_markdown"] = "transcript-semantic-readable-impact-report.md"
            manifest["mcp_transcript_semantic_readable_impact_report_args"] = "mcp-transcript-semantic-readable-impact-report.args.json"
            manifest["transcript_semantic_correction_readable_impact_summary"] = {"status": status, "accepted_decision_count": len(accepted), "required_readable_residual_total": required_residual_total, "updated_at": result["updated_at"]}
            write_json(root / "manifest.json", manifest)
    return result

def _semantic_completion_gate(
    *,
    legacy_status: str,
    legacy_next_action_key: str,
    complete_statuses: set[str],
    candidate_discovery_complete: bool,
    unadjudicated_candidate_count: int,
    review_required_count: int,
    artifact_identity_current: bool,
    corrected_transcript_exists: bool,
    quality_gate_ok: bool,
) -> dict[str, Any]:
    """Return the additive production gate without rewriting legacy phases."""

    blockers: list[dict[str, Any]] = []

    def add(
        key: str,
        next_action_key: str,
        action_kind: str,
        detail: str,
    ) -> None:
        blockers.append(
            {
                "key": key,
                "next_action_key": next_action_key,
                "action_kind": action_kind,
                "detail": detail,
            }
        )

    phase_complete = legacy_status in complete_statuses
    if not phase_complete:
        add(
            "legacy_artifact_phase_incomplete",
            legacy_next_action_key or "inspect_bundle",
            "machine",
            f"artifact phase is {legacy_status}",
        )
    if not candidate_discovery_complete:
        add(
            "candidate_discovery_incomplete",
            "run_candidate_discovery",
            "machine",
            "the retained transcript has not been covered by a hash-bound discovery scan",
        )
    if int(unadjudicated_candidate_count or 0) > 0:
        add(
            "semantic_candidates_unadjudicated",
            "validate_current_pack",
            "machine",
            f"{int(unadjudicated_candidate_count)} current-pack candidates have no adjudication",
        )
    if int(review_required_count or 0) > 0:
        add(
            "semantic_candidates_require_human_review",
            "review_candidates",
            "human",
            f"{int(review_required_count)} candidates still require human review",
        )
    if not artifact_identity_current:
        add(
            "semantic_artifact_identity_not_current",
            "reimport_or_revalidate_current_pack",
            "machine",
            "validation or closure identity is missing or stale",
        )
    if not corrected_transcript_exists:
        add(
            "formal_corrected_transcript_missing",
            "run_closure",
            "machine",
            "source-arbitrated-transcript.json does not exist",
        )
    if not quality_gate_ok:
        add(
            "transcript_quality_gate_not_passed",
            "run_transcript_quality_gate",
            "machine",
            "transcript-quality-gate.json is missing or failed",
        )

    # Existing incomplete phases retain their established action.  Once the
    # artifact phase looks complete, the first strict blocker becomes the
    # mandatory action.  This separates compatibility from acceptance safety.
    primary = blockers[0] if blockers else {}
    if not phase_complete:
        required_next_action_key = legacy_next_action_key or "inspect_bundle"
        required_action_kind = (
            "human"
            if required_next_action_key in {"review_candidates"}
            else "machine"
        )
    else:
        required_next_action_key = str(primary.get("next_action_key") or "none")
        required_action_kind = str(primary.get("action_kind") or "none")
    gate_ok = phase_complete and not blockers
    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_completion_gate.v1",
        "status": "passed" if gate_ok else "blocked",
        "ok": gate_ok,
        "legacy_phase_complete": phase_complete,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "required_next_action_key": required_next_action_key,
        "required_action_kind": required_action_kind,
    }


def transcript_semantic_correction_status(bundle_dir: str | Path, *, write: bool = False) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    pack = _read_optional_json(root / "transcript-semantic-correction-pack.json")
    validation = _read_optional_json(root / "transcript-semantic-correction-validation.json")
    closure = _read_optional_json(root / "transcript-semantic-correction-closure.json")
    impact = _read_optional_json(root / "transcript-semantic-correction-impact-report.json")
    readable_impact = _read_optional_json(root / "transcript-semantic-readable-impact-report.json")
    summary_impact = _read_optional_json(root / "transcript-semantic-summary-impact-report.json")
    quality_gate = _read_optional_json(root / "transcript-quality-gate.json")
    decision_ledger = _read_optional_json(root / "transcript-semantic-correction-decision-ledger.json")
    cumulative_accepted = [row for row in (decision_ledger.get("decisions") if isinstance(decision_ledger, dict) else []) or [] if isinstance(row, dict)]
    corrected_transcript_path = _bundle_path(root, manifest.get("corrected_transcript_json") or manifest.get("source_arbitrated_transcript_json") or "source-arbitrated-transcript.json")
    transaction = _semantic_transaction_status(root)
    llm_draft = _llm_draft_status(root, manifest)
    candidate_discovery = _candidate_discovery_status(root, manifest)
    status, next_action_key = _status_from_artifacts(pack, validation, closure, impact, readable_impact, summary_impact)
    review_count = _semantic_review_count(root, validation)
    artifact_identity = _semantic_artifact_identity_status(root, pack, validation, closure)
    discovery_scan = semantic_discovery_scan_state(root)
    candidate_resolution = semantic_candidate_resolution(root)
    detail_summary = _status_detail_summary(pack, validation, root)
    if isinstance(validation, dict) and validation and not artifact_identity["current"]:
        status, next_action_key = "stale_validation_pack", "reimport_or_revalidate_current_pack"
    review_closure_summary = _semantic_review_closure_summary(root, validation, detail_summary)
    accepted_decision_count = len(cumulative_accepted) if cumulative_accepted else (
        int(validation.get("accepted_decision_count") or 0) if isinstance(validation, dict) else 0
    )
    corrected_transcript_exists = corrected_transcript_path.exists()
    # Only the shared hash-bound scan projection may declare discovery
    # complete.  Manifest summaries and empty/no-segment labels are not proof.
    candidate_discovery_complete = bool(discovery_scan.get("complete"))
    unadjudicated_candidate_count = int(
        candidate_resolution.get("unadjudicated_candidate_count") or 0
    )
    quality_gate_status = (
        str(quality_gate.get("status") or "missing")
        if isinstance(quality_gate, dict)
        else "missing"
    )
    quality_gate_ok = bool(
        isinstance(quality_gate, dict)
        and quality_gate
        and quality_gate.get("ok") is True
    )
    complete_statuses = {"impact_passed", "no_candidates", "arbitrated_no_change"}
    # `status` and `next_action_key` are the legacy artifact-phase API.  Keep
    # them stable so existing repair queues can still select build/validate/
    # closure/impact actions.  The stricter production-completion contract is
    # additive and fail-closed through `completion_gate`.
    completion_gate = _semantic_completion_gate(
        legacy_status=status,
        legacy_next_action_key=next_action_key,
        complete_statuses=complete_statuses,
        candidate_discovery_complete=candidate_discovery_complete,
        unadjudicated_candidate_count=unadjudicated_candidate_count,
        review_required_count=review_count,
        artifact_identity_current=artifact_identity.get("current") is True,
        corrected_transcript_exists=corrected_transcript_exists,
        quality_gate_ok=quality_gate_ok,
    )
    required_next_action_key = str(
        completion_gate.get("required_next_action_key") or next_action_key
    )
    ui_summary = _semantic_correction_ui_summary(
        pack,
        validation,
        closure,
        detail_summary,
        candidate_discovery,
        review_closure_summary,
        status=status,
        next_action_key=next_action_key,
        readable_impact=readable_impact,
        summary_impact=summary_impact,
    )
    if cumulative_accepted:
        ui_summary["accepted_decision_count"] = len(cumulative_accepted)
        ui_summary["accepted_decision_type_counts"] = _count_values(row.get("correction_type") for row in cumulative_accepted)
    status_ok = bool(completion_gate.get("ok"))
    result = {
        "schema": STATUS_SCHEMA,
        "bundle_dir": str(root),
        "status": status,
        "ok": status_ok,
        "completion_gate": completion_gate,
        "required_next_action_key": required_next_action_key,
        "required_action_kind": str(
            completion_gate.get("required_action_kind") or "none"
        ),
        "candidate_count": int(pack.get("candidate_count") or 0) if isinstance(pack, dict) else 0,
        "accepted_decision_count": accepted_decision_count,
        "arbitrated_no_change_count": int(validation.get("arbitrated_no_change_count") or 0) if isinstance(validation, dict) else 0,
        "review_required_count": review_count,
        "adjudicated_candidate_count": int(candidate_resolution.get("adjudicated_candidate_count") or 0),
        "unadjudicated_candidate_count": unadjudicated_candidate_count,
        "unadjudicated_candidate_ids": candidate_resolution.get("unadjudicated_candidate_ids") or [],
        "artifact_identity": artifact_identity,
        "final_residual_error_total": int(impact.get("final_residual_error_total") or 0) if isinstance(impact, dict) else 0,
        "readable_impact_status": str(readable_impact.get("status") or "missing") if isinstance(readable_impact, dict) else "missing",
        "readable_required_residual_total": int(readable_impact.get("required_readable_residual_total") or 0) if isinstance(readable_impact, dict) else 0,
        "closure_status": str(closure.get("status") or "missing") if isinstance(closure, dict) else "missing",
        "closure_ok": bool(closure.get("ok")) if isinstance(closure, dict) else False,
        "closure_applied_correction_count": int(closure.get("applied_correction_count") or 0) if isinstance(closure, dict) else 0,
        "closure_changed_segment_count": int(closure.get("changed_segment_count") or 0) if isinstance(closure, dict) else 0,
        "transaction_status": transaction["status"],
        "transaction_id": transaction["transaction_id"],
        "transaction_journal": transaction["journal"],
        "transaction_restore_plan": transaction["restore_plan"],
        "latest_transaction_status": transaction["latest_status"],
        "latest_transaction_journal": transaction["latest_journal"],
        "corrected_transcript_exists": corrected_transcript_exists,
        "corrected_transcript_path": str(corrected_transcript_path),
        "summary_impact_status": str(summary_impact.get("status") or "missing") if isinstance(summary_impact, dict) else "missing",
        "summary_impact_ok": bool(summary_impact.get("ok")) if isinstance(summary_impact, dict) else False,
        "summary_absorption_rate": float(summary_impact.get("summary_absorption_rate") or 0.0) if isinstance(summary_impact, dict) else 0.0,
        "summary_residual_original_total": int(summary_impact.get("summary_residual_original_total") or 0) if isinstance(summary_impact, dict) else 0,
        "llm_draft_status": llm_draft["status"],
        "llm_draft_next_action": llm_draft["next_action_key"],
        "llm_draft_decision_count": llm_draft["decision_count"],
        "llm_draft_error": llm_draft["error"],
        "llm_draft_artifacts": llm_draft["artifacts"],
        "candidate_discovery_status": candidate_discovery["status"],
        "candidate_discovery_complete": candidate_discovery_complete,
        "candidate_discovery_scan": discovery_scan,
        "candidate_discovery_next_action": candidate_discovery["next_action_key"],
        "candidate_discovery_segment_count": candidate_discovery["segment_count"],
        "candidate_discovery_suggestion_count": candidate_discovery["suggestion_count"],
        "candidate_discovery_imported_candidate_count": candidate_discovery["imported_candidate_count"],
        "candidate_discovery_skipped_count": candidate_discovery["skipped_count"],
        "candidate_discovery_artifacts": candidate_discovery["artifacts"],
        "quality_gate_status": quality_gate_status,
        "quality_gate_ok": quality_gate_ok,
        "quality_layers": quality_gate.get("quality_layers") if isinstance(quality_gate, dict) else {},
        "candidate_type_counts": detail_summary["candidate_type_counts"],
        "risk_level_counts": detail_summary["risk_level_counts"],
        "candidate_group_count": detail_summary["candidate_group_count"],
        "candidate_group_preview": detail_summary["candidate_group_preview"],
        "evidence_source_counts": detail_summary["evidence_source_counts"],
        "validation_rejection_reason_counts": detail_summary["validation_rejection_reason_counts"],
        "review_required_items": detail_summary["review_required_items"],
        "review_required_preview": detail_summary["review_required_preview"],
        "semantic_attention_items": detail_summary["semantic_attention_items"],
        "semantic_attention_preview": detail_summary["semantic_attention_preview"],
        "source_vote_summary": detail_summary["source_vote_summary"],
        "review_closure_summary": review_closure_summary,
        "ui_summary": ui_summary,
        "chapter_risk_summary": detail_summary["chapter_risk_summary"],
        "next_action_key": next_action_key,
        "artifacts": _artifact_paths(root),
        "commands": _status_commands(root),
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_semantic_correction_status", timeout_seconds=1.0):
            write_json(root / "transcript-semantic-correction-status.json", result)
            (root / "transcript-semantic-correction-status.md").write_text(_render_status_markdown(result), encoding="utf-8")
            write_json(root / "mcp-transcript-semantic-correction-status.args.json", {"bundle_dir": str(root), "write": True})
            manifest["transcript_semantic_correction_status_json"] = "transcript-semantic-correction-status.json"
            manifest["transcript_semantic_correction_status_markdown"] = "transcript-semantic-correction-status.md"
            manifest["mcp_transcript_semantic_correction_status_args"] = "mcp-transcript-semantic-correction-status.args.json"
            manifest["transcript_semantic_correction_status_summary"] = {"status": status, "candidate_count": result["candidate_count"], "accepted_decision_count": result["accepted_decision_count"], "arbitrated_no_change_count": result["arbitrated_no_change_count"], "review_required_count": result["review_required_count"], "adjudicated_candidate_count": result["adjudicated_candidate_count"], "unadjudicated_candidate_count": result["unadjudicated_candidate_count"], "final_residual_error_total": result["final_residual_error_total"], "readable_impact_status": result["readable_impact_status"], "readable_required_residual_total": result["readable_required_residual_total"], "closure_status": result["closure_status"], "closure_applied_correction_count": result["closure_applied_correction_count"], "corrected_transcript_exists": result["corrected_transcript_exists"], "summary_impact_status": result["summary_impact_status"], "summary_absorption_rate": result["summary_absorption_rate"], "summary_residual_original_total": result["summary_residual_original_total"], "llm_draft_status": result["llm_draft_status"], "llm_draft_next_action": result["llm_draft_next_action"], "llm_draft_decision_count": result["llm_draft_decision_count"], "candidate_discovery_status": result["candidate_discovery_status"], "candidate_discovery_complete": result["candidate_discovery_complete"], "candidate_discovery_next_action": result["candidate_discovery_next_action"], "candidate_discovery_segment_count": result["candidate_discovery_segment_count"], "candidate_discovery_suggestion_count": result["candidate_discovery_suggestion_count"], "candidate_discovery_imported_candidate_count": result["candidate_discovery_imported_candidate_count"], "quality_gate_status": result["quality_gate_status"], "quality_gate_ok": result["quality_gate_ok"], "candidate_type_counts": result["candidate_type_counts"], "candidate_group_count": result["candidate_group_count"], "validation_rejection_reason_counts": result["validation_rejection_reason_counts"], "source_vote_summary": result["source_vote_summary"], "ui_summary": result["ui_summary"], "chapter_risk_summary": result["chapter_risk_summary"][:8], "semantic_attention_preview": result["semantic_attention_preview"][:8], "next_action_key": next_action_key, "updated_at": result["updated_at"]}
            manifest["transcript_semantic_correction_status_summary"].update(
                {
                    "ok": result["ok"],
                    "completion_gate": result["completion_gate"],
                    "required_next_action_key": result["required_next_action_key"],
                    "required_action_kind": result["required_action_kind"],
                }
            )
            write_json(root / "manifest.json", manifest)
    return result


def _semantic_transaction_status(root: Path) -> dict[str, str]:
    runs_root = root / "runs" / "transcript-semantic-correction"
    journals: list[Path] = []
    if runs_root.is_dir():
        try:
            journals = [
                path
                for path in runs_root.glob("*/transaction-journal.json")
                if path.is_file()
            ]
        except OSError:
            journals = []
    journals.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    latest_path = journals[0] if journals else None
    latest = _read_optional_json(latest_path) if latest_path is not None else {}
    committed_path: Path | None = None
    committed: dict[str, Any] = {}
    for path in journals:
        value = _read_optional_json(path)
        if isinstance(value, dict) and str(value.get("status") or "") == "committed":
            committed_path = path
            committed = value
            break
    selected_path = committed_path or latest_path
    selected = committed if committed_path is not None else (latest if isinstance(latest, dict) else {})
    restore_value = str(selected.get("restore_plan") or "") if isinstance(selected, dict) else ""
    restore_path = root / restore_value if restore_value else None
    return {
        "status": str(selected.get("status") or "missing") if isinstance(selected, dict) else "missing",
        "transaction_id": str(selected.get("transaction_id") or "") if isinstance(selected, dict) else "",
        "journal": str(selected_path.resolve()) if selected_path is not None else "",
        "restore_plan": str(restore_path.resolve()) if restore_path is not None else "",
        "latest_status": str(latest.get("status") or "missing") if isinstance(latest, dict) else "missing",
        "latest_journal": str(latest_path.resolve()) if latest_path is not None else "",
    }



def _llm_draft_status(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("transcript_semantic_correction_llm_draft_summary") if isinstance(manifest.get("transcript_semantic_correction_llm_draft_summary"), dict) else {}
    codex_summary = manifest.get("transcript_semantic_correction_codex_draft_summary") if isinstance(manifest.get("transcript_semantic_correction_codex_draft_summary"), dict) else {}
    prompt_path = root / str(manifest.get("transcript_semantic_correction_llm_prompt_markdown") or "transcript-semantic-correction-llm-prompt.md")
    result_json_path = root / str(manifest.get("transcript_semantic_correction_result_llm_json") or "transcript-semantic-correction-result.llm.json")
    result_md_path = root / str(manifest.get("transcript_semantic_correction_result_llm_markdown") or "transcript-semantic-correction-result.llm.md")
    codex_json_path = root / str(manifest.get("transcript_semantic_correction_result_codex_json") or "transcript-semantic-correction-result.codex.json")
    codex_md_path = root / str(manifest.get("transcript_semantic_correction_result_codex_markdown") or "transcript-semantic-correction-result.codex.md")
    raw_path = root / str(manifest.get("transcript_semantic_correction_result_llm_raw_text") or "transcript-semantic-correction-result.llm.raw.txt")
    raw_status = str(summary.get("status") or "").strip()
    raw_codex_status = str(codex_summary.get("status") or "").strip()
    decision_count = int(summary.get("decision_count") or 0)
    error = str(summary.get("error") or "")
    if result_json_path.exists() or result_md_path.exists() or raw_status == "executed":
        status = "executed"
        next_action_key = "validate_llm_result"
    elif raw_path.exists() or raw_status == "model_output_parse_failed":
        status = "model_output_parse_failed"
        next_action_key = "retry_llm_or_manual_review"
    elif codex_json_path.exists() or codex_md_path.exists() or raw_codex_status:
        decision_count = int(codex_summary.get("decision_count") or 0)
        if raw_codex_status == "no_safe_draft_decisions" or decision_count <= 0:
            status = "codex_no_safe_draft_decisions"
            next_action_key = "review_candidates"
        else:
            status = "codex_draft_ready"
            next_action_key = "validate_result"
    elif raw_status and raw_status not in {"planned", "executed"}:
        status = raw_status
        next_action_key = "retry_llm_or_manual_review"
    elif prompt_path.exists() or raw_status == "planned":
        status = "prompt_ready"
        next_action_key = "execute_llm_or_use_codex"
    else:
        status = "not_planned"
        next_action_key = "run_llm_draft_preview"
    return {
        "status": status,
        "next_action_key": next_action_key,
        "decision_count": decision_count,
        "error": error,
        "prompt_exists": prompt_path.exists(),
        "result_json_exists": result_json_path.exists(),
        "result_markdown_exists": result_md_path.exists(),
        "codex_result_json_exists": codex_json_path.exists(),
        "codex_result_markdown_exists": codex_md_path.exists(),
        "raw_output_exists": raw_path.exists(),
        "artifacts": {
            "prompt_markdown": str(prompt_path),
            "result_json": str(result_json_path),
            "result_markdown": str(result_md_path),
            "codex_result_json": str(codex_json_path),
            "codex_result_markdown": str(codex_md_path),
            "raw_output": str(raw_path),
        },
    }

def _candidate_discovery_status(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("transcript_semantic_candidate_discovery_summary") if isinstance(manifest.get("transcript_semantic_candidate_discovery_summary"), dict) else {}
    llm_summary = manifest.get("transcript_semantic_candidate_discovery_llm_draft_summary") if isinstance(manifest.get("transcript_semantic_candidate_discovery_llm_draft_summary"), dict) else {}
    pack_path = root / str(manifest.get("transcript_semantic_candidate_discovery_pack_json") or "transcript-semantic-candidate-discovery-pack.json")
    prompt_path = root / str(manifest.get("transcript_semantic_candidate_discovery_prompt_markdown") or "transcript-semantic-candidate-discovery-prompt.md")
    template_path = root / str(manifest.get("transcript_semantic_candidate_discovery_template_json") or "transcript-semantic-candidate-discovery-template.json")
    llm_prompt_path = root / str(manifest.get("transcript_semantic_candidate_discovery_llm_prompt_markdown") or "transcript-semantic-candidate-discovery-llm-prompt.md")
    llm_json_path = root / str(manifest.get("transcript_semantic_candidate_suggestions_llm_json") or "transcript-semantic-candidate-suggestions.llm.json")
    llm_md_path = root / str(manifest.get("transcript_semantic_candidate_suggestions_llm_markdown") or "transcript-semantic-candidate-suggestions.llm.md")
    codex_md_path = root / str(manifest.get("transcript_semantic_candidate_suggestions_codex_markdown") or "transcript-semantic-candidate-suggestions.codex.md")
    llm_raw_path = root / str(manifest.get("transcript_semantic_candidate_suggestions_llm_raw_text") or "transcript-semantic-candidate-suggestions.llm.raw.txt")
    import_path = root / str(manifest.get("transcript_semantic_candidate_suggestions_import_json") or "transcript-semantic-candidate-suggestions-import.json")
    pack = _read_optional_json(pack_path)
    llm_payload = _read_optional_json(llm_json_path)
    codex_payload: dict[str, Any] = {}
    if codex_md_path.exists():
        try:
            codex_payload = _load_import(codex_md_path)
        except Exception:
            codex_payload = {}
    imported = _read_optional_json(import_path)
    segment_count = int(summary.get("segment_count") or 0)
    if isinstance(pack, dict):
        segment_count = int(pack.get("segment_count") or len(pack.get("segments") or []) or segment_count)
    suggestion_count = int(llm_summary.get("suggestion_count") or 0)
    if isinstance(llm_payload, dict):
        suggestions = llm_payload.get("suggestions") if isinstance(llm_payload.get("suggestions"), list) else []
        suggestion_count = int(llm_payload.get("suggestion_count") or len(suggestions) or suggestion_count)
    if isinstance(codex_payload, dict):
        codex_suggestions = codex_payload.get("suggestions") if isinstance(codex_payload.get("suggestions"), list) else []
        suggestion_count = max(suggestion_count, int(codex_payload.get("suggestion_count") or len(codex_suggestions) or 0))
    imported_count = int(imported.get("imported_candidate_count") or 0) if isinstance(imported, dict) else 0
    skipped_count = int(imported.get("skipped_count") or 0) if isinstance(imported, dict) else 0
    raw_status = str(summary.get("status") or "").strip()
    raw_llm_status = str(llm_summary.get("status") or "").strip()
    if import_path.exists() or imported:
        status = "imported" if imported_count > 0 else "no_candidates_imported"
        next_action_key = "run_llm_draft_preview" if imported_count > 0 else "review_or_accept_no_candidates"
    elif llm_json_path.exists() or llm_md_path.exists() or codex_md_path.exists() or raw_llm_status == "executed":
        status = "suggestions_ready" if suggestion_count > 0 else "no_suggestions"
        next_action_key = "import_candidate_suggestions" if suggestion_count > 0 else "review_or_accept_no_candidates"
    elif llm_raw_path.exists() or raw_llm_status == "model_output_parse_failed":
        status = "model_output_parse_failed"
        next_action_key = "retry_candidate_discovery_llm_or_manual_review"
    elif llm_prompt_path.exists() or raw_llm_status == "planned":
        status = "llm_prompt_ready"
        next_action_key = "execute_candidate_discovery_llm_or_use_codex"
    elif pack_path.exists() or prompt_path.exists() or raw_status:
        status = "prompt_ready" if segment_count > 0 else "no_segments_selected"
        next_action_key = "run_candidate_discovery_llm_preview" if segment_count > 0 else "review_or_accept_no_candidates"
    else:
        status = "not_planned"
        next_action_key = "run_candidate_discovery"
    return {
        "status": status,
        "next_action_key": next_action_key,
        "segment_count": segment_count,
        "suggestion_count": suggestion_count,
        "imported_candidate_count": imported_count,
        "skipped_count": skipped_count,
        "llm_error": str(llm_summary.get("error") or ""),
        "artifacts": {
            "pack_json": str(pack_path),
            "prompt_markdown": str(prompt_path),
            "template_json": str(template_path),
            "llm_prompt_markdown": str(llm_prompt_path),
            "llm_suggestions_json": str(llm_json_path),
            "llm_suggestions_markdown": str(llm_md_path),
            "codex_suggestions_markdown": str(codex_md_path),
            "llm_raw_output": str(llm_raw_path),
            "import_json": str(import_path),
        },
    }
def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("manifest.json must be a JSON object")
    return data


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.suffix.lower() == ".json" else None
    try:
        return read_json(path)
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return sha256_file(path)


def _semantic_artifact_identity_status(
    root: Path,
    pack: Any,
    validation: Any,
    closure: Any,
) -> dict[str, Any]:
    pack_path = root / "transcript-semantic-correction-pack.json"
    current_sha256 = _sha256_file(pack_path)
    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in (pack.get("candidates") if isinstance(pack, dict) else []) or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    issues: list[dict[str, Any]] = []
    recorded_sha256 = str(validation.get("pack_sha256") or "") if isinstance(validation, dict) else ""
    if recorded_sha256 and current_sha256 and recorded_sha256 != current_sha256:
        issues.append(
            {
                "key": "validation_pack_sha256_mismatch",
                "recorded": recorded_sha256,
                "current": current_sha256,
            }
        )
    anchored_decisions: list[dict[str, Any]] = []
    if isinstance(validation, dict):
        for key in ("accepted_decisions", "arbitrated_no_change_decisions"):
            anchored_decisions.extend(row for row in validation.get(key) or [] if isinstance(row, dict))
    for decision in anchored_decisions:
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = candidates.get(candidate_id)
        if not candidate:
            issues.append({"key": "validated_candidate_missing_from_current_pack", "candidate_id": candidate_id})
            continue
        decision_original = str(decision.get("original_text") or "")
        candidate_original = str(candidate.get("original_text") or "")
        if decision_original and candidate_original and decision_original != candidate_original:
            issues.append(
                {
                    "key": "validated_candidate_original_mismatch",
                    "candidate_id": candidate_id,
                    "validated_original_text": decision_original,
                    "current_original_text": candidate_original,
                }
            )
    closure_sha256 = str(closure.get("pack_sha256") or "") if isinstance(closure, dict) else ""
    if closure_sha256 and current_sha256 and closure_sha256 != current_sha256:
        issues.append(
            {
                "key": "closure_pack_sha256_mismatch",
                "recorded": closure_sha256,
                "current": current_sha256,
            }
        )
    return {
        "current": not issues,
        "pack_path": str(pack_path),
        "current_pack_sha256": current_sha256,
        "validation_pack_sha256": recorded_sha256,
        "closure_pack_sha256": closure_sha256,
        "legacy_identity_fallback": bool(isinstance(validation, dict) and validation and not recorded_sha256),
        "anchored_decision_count": len(anchored_decisions),
        "issues": issues,
    }


def _bundle_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path



def _load_semantic_closure_cues(root: Path, manifest: dict[str, Any], timeline: list[Any]) -> list[Any]:
    """Load the richest stable upstream transcript before semantic corrections."""
    preferred_paths: list[Path] = []
    explicit = manifest.get("transcript_semantic_correction_base_json")
    if explicit:
        preferred_paths.append(_bundle_path(root, explicit))
    preferred_paths.extend(
        [
            root / "corrected-transcript.json",
            root / "agent-readable-transcript.json",
            root / "readable-transcript.json",
        ]
    )
    for path in preferred_paths:
        if path.name.lower() == "source-arbitrated-transcript.json" or not path.exists():
            continue
        cues = parse_transcript(path)
        if cues:
            return cues
    return _load_raw_correction_cues(root, manifest, timeline)

def _load_raw_correction_cues(root: Path, manifest: dict[str, Any], timeline: list[Any]) -> list[Any]:
    """Load the uncorrected transcript source used for semantic correction.

    Correction discovery and closure must not read source-arbitrated/corrected
    transcripts; otherwise rerunning the pack after closure consumes its own
    output and loses the original error evidence.
    """
    path = _raw_correction_source_path(root, manifest)
    if path is not None:
        return parse_transcript(path)
    return _timeline_cues(timeline)


def _raw_correction_source_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    """Resolve the immutable source used by discovery without consuming outputs."""

    keys = ["normalized_transcript_json", "raw_transcript_json", "asr_transcript_json", "transcript_json"]
    names = ["normalized-transcript.json", "raw-asr-output.json", "transcript.json"]
    for key in keys:
        value = manifest.get(key)
        if value:
            path = _bundle_path(root, value)
            if key == "transcript_json" and _looks_corrected_transcript_path(path):
                continue
            if path.is_file():
                return path.resolve()
    for name in names:
        path = root / name
        if path.is_file():
            return path.resolve()
    return None

def _looks_corrected_transcript_path(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in ("corrected", "source-arbitrated", "readable"))


def _load_best_cues(root: Path, manifest: dict[str, Any]) -> list[Any]:
    keys = ["source_arbitrated_transcript_json", "human_corrected_transcript_json", "llm_corrected_transcript_json", "corrected_transcript_json", "normalized_transcript_json", "transcript_json"]
    names = ["source-arbitrated-transcript.json", "normalized-transcript.json", "transcript.json"]
    for key in keys:
        value = manifest.get(key)
        if value:
            path = _bundle_path(root, value)
            if path.exists():
                return parse_transcript(path)
    for name in names:
        path = root / name
        if path.exists():
            return parse_transcript(path)
    return []



def _timeline_cues(timeline: list[Any]) -> list[Any]:
    cues: list[Any] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        text = _flatten_text(item.get("transcript") or item.get("subtitle") or item.get("caption") or item.get("original_subtitle") or item.get("text"))
        if not text:
            continue
        start = _float(item.get("start", item.get("start_seconds", 0.0)))
        end = _float(item.get("end", item.get("end_seconds", start)), start)
        cues.append(type("Cue", (), {"start": start, "end": end, "text": text})())
    return cues

def _build_candidates(
    cues: list[Any],
    timeline: list[Any],
    *,
    sidecar_sources: list[dict[str, Any]] | None = None,
    metadata_evidence: list[dict[str, Any]] | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    sidecar_sources = sidecar_sources or []
    metadata_evidence = metadata_evidence or []
    for idx, cue in enumerate(cues):
        text = str(getattr(cue, "text", "") or "").strip()
        if not text:
            continue
        start = _float(getattr(cue, "start", 0.0))
        end = _float(getattr(cue, "end", 0.0), start)
        tl_item, timeline_alignment = _timeline_match_for_cue(
            timeline,
            cue,
            start,
            end,
        )
        evidence = _dedupe_evidence([
            *_evidence_for_cue(
                cue,
                tl_item,
                segment_index=idx,
                timeline_alignment=timeline_alignment,
            ),
            *_sidecar_evidence_for_cue(cue, sidecar_sources),
            *metadata_evidence,
        ])
        for candidate in _candidate_rows_for_text(idx, cue, text, evidence):
            key = (candidate["original_text"].lower(), str(candidate.get("candidate_text") or "").lower(), candidate["correction_type"])
            if key in seen:
                continue
            seen.add(key)
            candidate["candidate_id"] = f"semcorr-{len(rows)+1:04d}"
            rows.append(candidate)
            if limit and len(rows) >= limit:
                return rows
    return rows


def _attach_candidate_context(
    candidates: list[dict[str, Any]],
    cues: list[Any],
    *,
    chapters: list[dict[str, Any]],
) -> None:
    """Attach bounded, read-only context without expanding the editable span."""

    labels = speaker_label_map(cues)
    contexts: dict[int, dict[str, Any]] = {}
    candidate_segments = {
        int(row["segment_index"])
        for row in candidates
        if isinstance(row.get("segment_index"), int)
    }
    for candidate in candidates:
        source_index = candidate.get("segment_index")
        if not isinstance(source_index, int) or not (0 <= source_index < len(cues)):
            continue
        if source_index not in contexts:
            contexts[source_index] = _bounded_semantic_context(
                cues,
                source_index,
                speaker_labels=labels,
            )
        candidate["context_window"] = _isolate_candidate_context(
            contexts[source_index],
            source_index=source_index,
            candidate_segments=candidate_segments,
        )
        candidate.update(speaker_payload(cues[source_index], labels))
        chapter = _chapter_for_semantic_time(
            chapters,
            _float(candidate.get("start", 0.0)),
            _float(candidate.get("end", candidate.get("start", 0.0))),
        )
        candidate["chapter_index"] = int(chapter.get("chapter_index") or 0)
        candidate["chapter_title"] = str(chapter.get("chapter_title") or "unassigned")
        candidate["chapter_time_range"] = str(chapter.get("chapter_time_range") or "")
        candidate["edit_contract"] = {
            "target_segment_index": source_index,
            "target_original_text": str(candidate.get("original_text") or ""),
            "only_target_span_editable": True,
            "context_read_only": True,
        }


def _isolate_candidate_context(
    context: dict[str, Any],
    *,
    source_index: int,
    candidate_segments: set[int],
) -> dict[str, Any]:
    """Keep sibling candidate spans out of a model task's editable context."""

    isolated = dict(context)
    for key in ("previous", "next"):
        rows: list[dict[str, Any]] = []
        for item in context.get(key) or []:
            row = dict(item) if isinstance(item, dict) else {}
            index = row.get("source_segment_index")
            if isinstance(index, int) and index != source_index and index in candidate_segments:
                text = str(row.get("text") or "")
                row["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
                row["text"] = ""
                row["withheld_reason"] = "neighbor_has_separate_semantic_candidate"
            rows.append(row)
        isolated[key] = rows
    isolated["target"] = dict(context.get("target") or {})
    return isolated


def _bounded_semantic_context(
    cues: list[Any],
    source_index: int,
    *,
    speaker_labels: dict[str, str],
) -> dict[str, Any]:
    selected = {source_index}
    offsets: list[int] = []
    for distance in range(1, SEMANTIC_CONTEXT_MAX_CUES_PER_SIDE + 1):
        offsets.extend((-distance, distance))
    for offset in offsets:
        index = source_index + offset
        if not (0 <= index < len(cues)):
            continue
        if not str(getattr(cues[index], "text", "") or "").strip():
            continue
        tentative = selected | {index}
        starts = [_float(getattr(cues[item], "start", 0.0)) for item in tentative]
        ends = [
            _float(getattr(cues[item], "end", starts[pos]), starts[pos])
            for pos, item in enumerate(tentative)
        ]
        if max(ends) - min(starts) <= SEMANTIC_CONTEXT_MAX_SECONDS:
            selected.add(index)
    ordered = sorted(selected)
    previous = [
        _semantic_context_cue(cues[index], index, speaker_labels)
        for index in ordered
        if index < source_index
    ]
    following = [
        _semantic_context_cue(cues[index], index, speaker_labels)
        for index in ordered
        if index > source_index
    ]
    starts = [_float(getattr(cues[index], "start", 0.0)) for index in ordered]
    ends = [
        _float(getattr(cues[index], "end", starts[pos]), starts[pos])
        for pos, index in enumerate(ordered)
    ]
    window_start = min(starts) if starts else 0.0
    window_end = max(ends) if ends else window_start
    return {
        "previous": previous,
        "target": _semantic_context_cue(cues[source_index], source_index, speaker_labels),
        "next": following,
        "start": window_start,
        "end": window_end,
        "duration_seconds": round(max(0.0, window_end - window_start), 6),
        "max_duration_seconds": SEMANTIC_CONTEXT_MAX_SECONDS,
        "max_cues_per_side": SEMANTIC_CONTEXT_MAX_CUES_PER_SIDE,
        "context_read_only": True,
    }


def _semantic_context_cue(
    cue: Any,
    source_index: int,
    speaker_labels: dict[str, str],
) -> dict[str, Any]:
    start = _float(getattr(cue, "start", 0.0))
    end = _float(getattr(cue, "end", start), start)
    return {
        "source_segment_index": source_index,
        "start": start,
        "end": end,
        "time_range": f"{format_timestamp(start)} - {format_timestamp(end)}",
        "text": str(getattr(cue, "text", "") or "")[:1200],
        **speaker_payload(cue, speaker_labels),
    }



def _assign_candidate_groups(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups_by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for candidate in candidates:
        key, canonical_hint = _candidate_group_key(candidate)
        group = groups_by_key.get(key)
        if group is None:
            group = {
                "candidate_group_id": f"semgroup-{len(ordered)+1:04d}",
                "canonical_hint": canonical_hint,
                "correction_type": str(candidate.get("correction_type") or "ordinary_word"),
                "correction_types": [],
                "risk_level": str(candidate.get("risk_level") or "unknown"),
                "candidate_ids": [],
                "variant_texts": [],
                "suggested_texts": [],
                "timeline_indexes": [],
                "evidence_ids": [],
                "evidence_source_types": [],
                "time_ranges": [],
                "needs_human_review": False,
                "reasons": [],
            }
            groups_by_key[key] = group
            ordered.append(group)
        if canonical_hint and _canonical_hint_quality(canonical_hint) > _canonical_hint_quality(str(group.get("canonical_hint") or "")):
            group["canonical_hint"] = canonical_hint
        candidate["candidate_group_id"] = group["candidate_group_id"]
        candidate["canonical_hint"] = group["canonical_hint"]
        _append_unique(group["correction_types"], str(candidate.get("correction_type") or "ordinary_word"))
        _append_unique(group["candidate_ids"], str(candidate.get("candidate_id") or ""))
        _append_unique(group["variant_texts"], _candidate_group_variant_text(candidate))
        _append_unique(group["suggested_texts"], str(candidate.get("suggested_text") or candidate.get("candidate_text") or ""))
        for index in candidate.get("timeline_indexes") or []:
            _append_unique(group["timeline_indexes"], int(index))
        for evidence_id in candidate.get("evidence_ids") or []:
            _append_unique(group["evidence_ids"], str(evidence_id))
        for source_type in candidate.get("evidence_source_types") or []:
            _append_unique(group["evidence_source_types"], str(source_type))
        _append_unique(group["time_ranges"], str(candidate.get("time_range") or ""))
        _append_unique(group["reasons"], str(candidate.get("reason") or ""))
        group["needs_human_review"] = bool(group.get("needs_human_review")) or bool(candidate.get("needs_human_review"))
        group["risk_level"] = _higher_risk(str(group.get("risk_level") or "unknown"), str(candidate.get("risk_level") or "unknown"))
    for group in ordered:
        group["candidate_count"] = len(group.get("candidate_ids") or [])
        group["evidence_count"] = len(group.get("evidence_ids") or [])
    return ordered


def _candidate_group_key(candidate: dict[str, Any]) -> tuple[str, str]:
    hint = _candidate_canonical_hint(candidate)
    key_text = _normalise_group_text(hint or str(candidate.get("original_text") or ""))
    if not key_text:
        key_text = str(candidate.get("candidate_id") or "unknown")
    return f"canonical:{key_text}", hint


def _candidate_canonical_hint(candidate: dict[str, Any]) -> str:
    for value in (candidate.get("suggested_text"), candidate.get("candidate_text")):
        text = str(value or "").strip()
        if text:
            return text
    for original, _key, mapped in _candidate_draft_correction_matches(candidate):
        corrected = str(mapped[0] or "").strip()
        if corrected:
            return corrected
    return str(candidate.get("original_text") or "").strip()


def _canonical_hint_quality(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    score = 1
    if re.search(r"[A-Z]", text):
        score += 2
    if " " not in text:
        score += 1
    if re.search(r"[\u4e00-\u9fff]", text):
        score -= 1
    return score


def _candidate_group_variant_text(candidate: dict[str, Any]) -> str:
    original = str(candidate.get("original_text") or "").strip()
    suggested = str(candidate.get("suggested_text") or candidate.get("candidate_text") or "").strip()
    if suggested and original and len(original) > len(suggested) + 6:
        phrases = _ascii_phrases_any_text(original)
        if phrases:
            return phrases[0]
    return original


def _ascii_phrases_any_text(text: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in ASCII_PHRASE_RE.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", match.group(0).strip())
        token_keys = [part.lower() for part in phrase.split()]
        if any(key in SUPPORT_STOP_TOKENS for key in token_keys):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def _normalise_group_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[`'\"“”‘’（）()\[\]{}<>，。！？；：、,.!?;:/\\|]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def _higher_risk(left: str, right: str) -> str:
    rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _append_unique(values: list[Any], value: Any) -> None:
    if value in {"", None}:
        return
    if value not in values:
        values.append(value)

def _candidate_rows_for_text(idx: int, cue: Any, text: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = _float(getattr(cue, "start", 0.0))
    end = _float(getattr(cue, "end", start), start)
    visual_text = " ".join(
        _strip_technical_visual_artifacts(str(item.get("text") or ""))
        for item in evidence
        # Model scene descriptions remain evidence for review, not literal
        # spoken words. Their schema keys must never become replacements.
        if item.get("source_type") in {"ocr", "structured_visual"}
    )
    raw_compact = _semantic_diff_compact(text)
    sidecar_parts = []
    for item in evidence:
        if item.get("source_type") not in {"platform_subtitle", "embedded_subtitle", "secondary_asr"}:
            continue
        part = str(item.get("text") or "")
        compact = _semantic_diff_compact(part)
        if len(compact) > max(1, len(raw_compact)) / 0.72:
            # Scope each source separately: an agreeing source must not hide
            # a conflict in another source. Reuse the existing window matcher.
            window = _best_semantic_diff_window(compact, raw_compact)
            offset = compact.find(window) if window else -1
            part = _semantic_diff_compact(part, lowercase=False)[offset:offset + len(window)] if offset >= 0 else ""
        if part and not (raw_compact and raw_compact in _semantic_diff_compact(part)):
            sidecar_parts.append(part)
    sidecar_text = " ".join(dict.fromkeys(sidecar_parts))
    tagger_text = " ".join(str(item.get("text") or "") for item in evidence if item.get("source_type") == "tagger")
    support_text = " ".join(part for part in (visual_text, sidecar_text, tagger_text) if part)
    if MOJIBAKE_RE.search(text):
        rows.append(_candidate(idx, start, end, text, _short_span(text), "ordinary_word", "high", evidence, reason="transcript_text_mojibake_or_decoding_error", candidate_text=_support_token_not_in_text(text, support_text)))
    for original, (corrected, rationale) in _domain_semantic_suspect_matches(text):
        lexicon_evidence = {
            "evidence_id": "domain_lexicon_" + hashlib.sha256(
                f"{original}\0{corrected}".encode("utf-8")
            ).hexdigest()[:16],
            "source_type": "explicit_domain_lexicon",
            "text": corrected,
            "original_variant": original,
            "canonical_term": corrected,
            "rationale": rationale,
            "version": "domain_semantic_suspect_corrections.v1",
        }
        rows.append(_candidate(idx, start, end, text, original, "ordinary_word", "medium", [*evidence, lexicon_evidence], reason="known_domain_semantic_suspect", candidate_text=corrected, has_conflict=True))
        rows[-1]["domain_semantic_rationale"] = rationale
        if original in DOMAIN_SEMANTIC_REVIEW_ONLY_VARIANTS:
            rows[-1]["needs_human_review"] = True
            rows[-1]["risk_level"] = "high"
            rows[-1]["candidate_only"] = True
            rows[-1]["automatic_application_allowed"] = False
    for num in _fact_value_markers(text)[:3]:
        if _is_generic_chinese_quantifier(num) and not _generic_quantifier_has_cross_source_difference(
            num,
            evidence,
        ):
            continue
        rows.append(_candidate(idx, start, end, text, num, "number", "high", evidence, reason="contains_number_or_amount"))
    if ACTION_HINT_RE.search(text):
        suggested = _visual_conflict_text(text, " ".join(part for part in (visual_text, tagger_text) if part))
        rows.append(_candidate(idx, start, end, text, _short_span(text), "action", "medium", evidence, reason="action_or_step_word_in_transcript", candidate_text=suggested))
    compound_spaced_terms = _compound_spaced_ascii_terms(text)
    for item in compound_spaced_terms[:2]:
        rows.append(_candidate(idx, start, end, text, item, "proper_noun", "medium", evidence, reason="compound_spaced_tool_or_proper_noun", candidate_text=_compound_spaced_suggestion(item)))
    for item in ODD_SPACING_RE.findall(text)[:2]:
        if any(item.lower() in compound.lower() for compound in compound_spaced_terms):
            continue
        rows.append(_candidate(idx, start, end, text, item, "proper_noun", "medium", evidence, reason="odd_spaced_letters_or_acronym", candidate_text=_spaced_letters_suggestion(item, text, support_text)))
    for item in _ascii_phrases_in_chinese_text(text)[:2]:
        suggested = _ascii_support_canonical_suggestion(item, support_text)
        rows.append(_candidate(idx, start, end, text, item, "proper_noun", "medium", evidence, reason="ascii_phrase_or_tool_name_in_chinese_transcript", candidate_text=suggested, has_conflict=bool(suggested)))
    for item in _ascii_tokens_in_chinese_text(text)[:3]:
        rows.append(_candidate(idx, start, end, text, item, "proper_noun", "medium", evidence, reason="ascii_tool_or_proper_noun_in_chinese_transcript", candidate_text=""))
    support_concept = _support_concept_phrase_not_in_text(text, support_text)
    if support_concept and _looks_deictic_or_low_information(text):
        rows.append(_candidate(idx, start, end, text, _short_span(text), "concept", "medium", evidence, reason="deictic_or_low_information_transcript_with_support_concept", candidate_text=support_concept, has_conflict=True))
    if FILLER_RE.search(text) or _looks_fragmented(text):
        rows.append(_candidate(idx, start, end, text, _short_span(text), "ordinary_word", "medium", evidence, reason="fragmented_or_semantically_weak_phrase", candidate_text=_support_token_not_in_text(text, support_text)))
    ordinary_conflict = _ordinary_subtitle_diff_candidate(text, sidecar_text)
    if ordinary_conflict:
        span, suggested = ordinary_conflict
        sidecar_types = {str(item.get("source_type") or "") for item in evidence}
        reason = "ordinary_word_conflict_between_dual_asr" if "secondary_asr" in sidecar_types else "ordinary_word_conflict_between_asr_and_subtitle"
        rows.append(_candidate(idx, start, end, text, span, "ordinary_word", "medium", evidence, reason=reason, candidate_text=suggested, has_conflict=True))
    visual_ordinary_conflict = _ordinary_support_diff_candidate(text, visual_text)
    if visual_ordinary_conflict:
        span, suggested = visual_ordinary_conflict
        rows.append(_candidate(idx, start, end, text, span, "ordinary_word", "medium", evidence, reason="ordinary_word_conflict_between_asr_and_visual_text", candidate_text=suggested, has_conflict=True))
    tagger_ordinary_conflict = _ordinary_support_diff_candidate(text, tagger_text)
    if tagger_ordinary_conflict:
        span, suggested = tagger_ordinary_conflict
        rows.append(_candidate(idx, start, end, text, span, "ordinary_word", "medium", evidence, reason="ordinary_word_conflict_between_asr_and_tagger", candidate_text=suggested, has_conflict=True))
    boundary_kind = _punctuation_or_boundary_kind(text, start=start, end=end)
    if boundary_kind:
        rows.append(_candidate(idx, start, end, text, text, boundary_kind, "high", evidence, reason="punctuation_or_segment_boundary_uncertain", candidate_text=""))
    # Page title/description are useful support evidence, but they are often
    # broader than the current spoken segment. Do not let metadata alone create
    # mismatch candidates, otherwise titles flood the review queue.
    conflict_sources = [("visual_text_differs_from_transcript", visual_text), ("subtitle_text_differs_from_transcript", sidecar_text), ("tagger_text_differs_from_transcript", tagger_text)]
    for reason, source_text in conflict_sources:
        conflict_text = _visual_conflict_text(text, source_text)
        if conflict_text:
            rows.append(_candidate(idx, start, end, text, _suspicious_span(text, conflict_text), _infer_correction_type(conflict_text), "medium", evidence, reason=reason, candidate_text=conflict_text, has_conflict=True))
    return rows


def _is_generic_chinese_quantifier(value: str) -> bool:
    return bool(GENERIC_CHINESE_QUANTIFIER_RE.fullmatch(str(value or "").replace(" ", "")))


def _generic_quantifier_has_cross_source_difference(
    value: str,
    evidence: list[dict[str, Any]],
) -> bool:
    compact = str(value or "").replace(" ", "").lower()
    unit = compact[-1:] if compact else ""
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_type") or "") in {"", "asr_or_subtitle", "page_metadata"}:
            continue
        for marker in _fact_value_markers(str(item.get("text") or "")):
            marker_compact = marker.replace(" ", "").lower()
            if marker_compact != compact and unit and marker_compact.endswith(unit):
                return True
    return False


def _domain_semantic_suspect_matches(text: str) -> list[tuple[str, tuple[str, str]]]:
    rows: list[tuple[str, tuple[str, str]]] = []
    matched_ranges: list[tuple[int, int]] = []
    for original in sorted(DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS, key=len, reverse=True):
        start = 0
        while True:
            index = text.find(original, start)
            if index < 0:
                break
            end = index + len(original)
            if not any(index >= left and end <= right for left, right in matched_ranges):
                matched_ranges.append((index, end))
                rows.append((original, DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS[original]))
            start = end
    return rows


def _candidate(
    idx: int,
    start: float,
    end: float,
    full_text: str,
    span: str,
    kind: str,
    risk: str,
    evidence: list[dict[str, Any]],
    *,
    reason: str,
    candidate_text: str = "",
    has_conflict: bool = False,
) -> dict[str, Any]:
    original = span.strip() or full_text[:80]
    suggested = candidate_text.strip()
    row = {
        "candidate_id": "",
        "segment_index": idx,
        "start": start,
        "end": end,
        "time_range": f"{format_timestamp(start)} - {format_timestamp(end)}",
        "correction_type": kind,
        "risk_level": risk,
        "original_text": original,
        "candidate_text": suggested,
        "suggested_text": suggested,
        "context_text": full_text,
        "reason": reason,
        "has_conflict": bool(has_conflict),
        "needs_human_review": kind in HIGH_RISK_TYPES or risk == "high",
        "evidence": evidence,
        "evidence_ids": [item["evidence_id"] for item in evidence],
        "evidence_source_types": sorted({str(item.get("source_type")) for item in evidence if item.get("source_type")}),
        "source_support_summary": _source_support_summary(original, suggested, evidence),
        "timeline_indexes": sorted({int(item["timeline_index"]) for item in evidence if str(item.get("timeline_index", "")).isdigit()}),
    }
    row.update(_candidate_llm_review_metadata(row))
    return row



def _candidate_llm_review_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    reason = str(candidate.get("reason") or "")
    source_types = {str(item) for item in candidate.get("evidence_source_types") or [] if str(item)}
    candidate_text = str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    original_text = str(candidate.get("original_text") or "").strip()
    summary = candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {}
    supports_candidate = {str(item) for item in summary.get("supports_candidate") or [] if str(item)}
    strong_candidate_sources = {str(item) for item in summary.get("strong_candidate_sources") or [] if str(item)}
    has_external_support = bool((supports_candidate - {"asr_or_subtitle", "page_metadata"}) or (source_types & LLM_SUPPORT_SOURCE_TYPES))
    has_source_conflict = bool(candidate.get("has_conflict") or summary.get("has_source_conflict") or reason in LLM_CONFLICT_REASONS)
    has_candidate_delta = bool(candidate_text and _normalize_compact(candidate_text) != _normalize_compact(original_text))
    has_known_term_signal = reason in {"compound_spaced_tool_or_proper_noun", "odd_spaced_letters_or_acronym", "ascii_phrase_or_tool_name_in_chinese_transcript", "ascii_tool_or_proper_noun_in_chinese_transcript", "transcript_text_mojibake_or_decoding_error"}

    eligible = False
    priority_class = "low_evidence_heuristic"
    defer_reason = "needs_conflicting_external_evidence"
    if has_candidate_delta and has_source_conflict:
        eligible = True
        priority_class = "source_conflict"
        defer_reason = ""
    elif has_candidate_delta and (strong_candidate_sources or (source_types & {"ocr", "structured_visual", "visual_understanding", "temporal_visual", "human_note"})):
        eligible = True
        priority_class = "strong_visual_or_human_evidence"
        defer_reason = ""
    elif has_candidate_delta and source_types & {"platform_subtitle", "embedded_subtitle", "secondary_asr", "tagger"}:
        eligible = True
        priority_class = "subtitle_or_tagger_evidence"
        defer_reason = ""
    elif has_candidate_delta and has_known_term_signal and has_external_support:
        eligible = True
        priority_class = "known_term_with_support"
        defer_reason = ""
    elif reason not in LOW_EVIDENCE_HEURISTIC_REASONS and has_candidate_delta:
        eligible = True
        priority_class = "candidate_delta"
        defer_reason = ""
    return {
        "llm_review_eligible": eligible,
        "llm_review_priority_class": priority_class,
        "llm_review_defer_reason": defer_reason,
    }


def _candidate_llm_review_eligible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("llm_review_eligible"))



def _source_support_summary(original_text: str, candidate_text: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    from .evidence_lineage import aggregate_source_votes

    original_norm = _normalize_compact(original_text)
    candidate_norm = _normalize_compact(candidate_text)
    supports_original: list[str] = []
    supports_candidate: list[str] = []
    neutral: list[str] = []
    votes: list[dict[str, Any]] = []
    candidate_weight = 0
    original_weight = 0
    neutral_weight = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("source_type") or "unknown")
        weight = _source_reliability_weight(source_type)
        text = str(item.get("text") or "")
        text_norm = _normalize_compact(text)
        vote = "neutral"
        if candidate_norm and candidate_norm in text_norm:
            vote = "supports_candidate"
            candidate_weight += weight
            _append_unique(supports_candidate, source_type)
        elif original_norm and (original_norm in text_norm or source_type == "asr_or_subtitle"):
            vote = "supports_original"
            original_weight += weight
            _append_unique(supports_original, source_type)
        else:
            neutral_weight += weight
            _append_unique(neutral, source_type)
        votes.append({
            "source_type": source_type,
            "source_weight": weight,
            "evidence_id": str(item.get("evidence_id") or ""),
            "vote": vote,
            "text_excerpt": text[:160],
        })
    lineage = aggregate_source_votes(evidence, votes)
    candidate_weight = lineage['candidate_weight']
    original_weight = lineage['original_weight']
    neutral_weight = lineage['neutral_weight']
    strong_candidate_sources = sorted(set(supports_candidate) & _strong_source_types())
    strong_original_sources = sorted((set(supports_original) - {"asr_or_subtitle"}) & _strong_source_types())
    conflict = bool(set(supports_candidate) and (set(supports_original) - {"asr_or_subtitle"}))
    margin = candidate_weight - original_weight
    if not supports_candidate:
        dominant = "original" if supports_original else "unknown"
    elif conflict and abs(margin) < 25:
        dominant = "conflict_needs_review"
    elif margin > 0:
        dominant = "candidate"
    elif margin < 0:
        dominant = "original"
    else:
        dominant = "tie_needs_review" if conflict else "candidate"
    needs_review = bool(conflict and (dominant in {"conflict_needs_review", "tie_needs_review"} or strong_original_sources))
    if lineage['intra_group_conflict_count'] or lineage['unresolved_group_count']:
        # Keep the direction of strong counter-evidence visible. A winning
        # candidate, however, cannot resolve its own provenance conflict.
        if dominant != 'original':
            dominant = 'conflict_needs_review'
        needs_review = True
    source_set = sorted(set(supports_candidate + supports_original + neutral))
    return {
        "supports_candidate": supports_candidate,
        "supports_original": supports_original,
        "neutral": neutral,
        "candidate_weight": candidate_weight,
        "original_weight": original_weight,
        "neutral_weight": neutral_weight,
        "weight_margin": margin,
        "dominant_side": dominant,
        "needs_review_by_source_vote": needs_review,
        "has_source_conflict": conflict,
        "strong_candidate_sources": strong_candidate_sources,
        "strong_original_sources": strong_original_sources,
        "source_reliability": {source: _source_reliability_weight(source) for source in source_set},
        "source_reliability_notes": {source: SOURCE_RELIABILITY_NOTES.get(source, "未登记来源，低权重处理。") for source in source_set},
        "votes": votes,
        "lineage": lineage,
    }


def _source_reliability_weight(source_type: str) -> int:
    return int(SOURCE_RELIABILITY_WEIGHTS.get(str(source_type or "unknown"), SOURCE_RELIABILITY_WEIGHTS["unknown"]))


def _strong_source_types() -> set[str]:
    return {"ocr", "structured_visual", "visual_understanding", "temporal_visual", "human_note"}


def _has_strong_source_opposition(candidate: dict[str, Any]) -> bool:
    summary = candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {}
    supports_candidate = {str(item) for item in summary.get("supports_candidate") or [] if str(item)}
    supports_original = {str(item) for item in summary.get("supports_original") or [] if str(item)} - {"asr_or_subtitle"}
    strong_sources = _strong_source_types()
    strong_original = supports_original & strong_sources
    strong_candidate = supports_candidate & strong_sources
    return bool(strong_original and not strong_candidate)


def _compound_spaced_ascii_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in COMPOUND_SPACED_ASCII_RE.finditer(text):
        term = re.sub(r"\s+", " ", match.group(0).strip())
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _compound_spaced_suggestion(span: str) -> str:
    parts = [part for part in re.split(r"\s+", span.strip()) if part]
    if not parts:
        return ""
    prefix = "".join(part.upper() for part in parts[:-1])
    tail = parts[-1]
    if not prefix:
        return tail
    return f"{prefix} {tail}"


def _ascii_phrases_in_chinese_text(text: str) -> list[str]:
    if not re.search(r"[\u4e00-\u9fff]", text):
        return []
    phrases: list[str] = []
    seen: set[str] = set()
    for match in ASCII_PHRASE_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(0).strip())
        token_keys = [part.lower() for part in phrase.split()]
        if any(key in SUPPORT_STOP_TOKENS for key in token_keys):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def _ascii_tokens_in_chinese_text(text: str) -> list[str]:
    if not re.search(r"[\u4e00-\u9fff]", text):
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for token in ASCII_TOKEN_RE.findall(text):
        key = token.lower()
        if key in SUPPORT_STOP_TOKENS or key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens


def _spaced_letters_suggestion(span: str, text: str, support_text: str) -> str:
    canonical = _ascii_support_canonical_suggestion(span, support_text)
    if canonical:
        return canonical
    supported = _support_token_not_in_text(text, support_text)
    if supported:
        return supported
    compact = re.sub(r"\s+", "", span or "")
    return compact.upper() if compact else ""


def _ascii_support_canonical_suggestion(span: str, support_text: str) -> str:
    compact = _normalize_compact(span)
    if not compact or not re.search(r"[a-z]", compact):
        return ""
    candidates = []
    candidates.extend(_ascii_phrases_any_text(support_text))
    candidates.extend(ASCII_TOKEN_RE.findall(support_text or ""))
    for candidate in candidates:
        value = re.sub(r"\s+", " ", str(candidate or "").strip())
        if not value:
            continue
        if _normalize_compact(value) == compact and value != span:
            return value
    return ""

def _timeline_match_for_cue(
    timeline: list[Any],
    cue: Any,
    start: float,
    end: float,
    *,
    window_seconds: float = TIMELINE_ALIGNMENT_WINDOW_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Match transcript evidence without aliasing unrelated numeric indexes.

    A transcript cue's zero-based list position and a timeline item's ``index``
    are independent namespaces.  A direct match is therefore allowed only when
    both records explicitly share a segment/frame identity.  Otherwise the
    best strict time overlap is used, with a small, recorded tolerance for
    timestamp rounding drift.  No unbounded nearest-frame fallback is allowed.
    """

    cue_ids = _cue_alignment_ids(cue)
    identity_matches: list[tuple[dict[str, Any], list[str]]] = []
    if cue_ids:
        for item in timeline:
            if not isinstance(item, dict):
                continue
            shared_ids = sorted(cue_ids & _timeline_alignment_ids(item))
            if shared_ids:
                identity_matches.append((item, shared_ids))
    if identity_matches:
        # A segment can legitimately own multiple frames.  Identity remains the
        # primary key; time only deterministically selects the closest member.
        selected_item, selected_ids = min(
            identity_matches,
            key=lambda row: _timeline_time_sort_key(row[0], start, end),
        )
        return selected_item, _timeline_alignment_record(
            selected_item,
            start,
            end,
            basis="shared_identity",
            window_seconds=window_seconds,
            shared_ids=selected_ids,
        )

    timed_items = [
        item
        for item in timeline
        if isinstance(item, dict) and _timeline_time_bounds(item) is not None
    ]
    overlapping = [
        item
        for item in timed_items
        if _timeline_intersects(item, start, end)
    ]
    if overlapping:
        selected = min(
            overlapping,
            key=lambda item: _timeline_time_sort_key(item, start, end),
        )
        return selected, _timeline_alignment_record(
            selected,
            start,
            end,
            basis="time_overlap",
            window_seconds=window_seconds,
        )

    bounded = [
        item
        for item in timed_items
        if _timeline_rounding_tolerance_allowed(item, start, end)
        and _timeline_distance_seconds(item, start, end) <= max(0.0, window_seconds)
    ]
    if bounded:
        selected = min(
            bounded,
            key=lambda item: _timeline_time_sort_key(item, start, end),
        )
        return selected, _timeline_alignment_record(
            selected,
            start,
            end,
            basis="bounded_time_window",
            window_seconds=window_seconds,
        )

    return None, {
        "status": "not_matched",
        "basis": "none",
        "cue_start": start,
        "cue_end": end,
        "max_window_seconds": max(0.0, window_seconds),
    }


def _timeline_overlap(timeline: list[Any], start: float, end: float) -> dict[str, Any] | None:
    """Compatibility wrapper for callers that only need a bounded time match."""

    item, _alignment = _timeline_match_for_cue(
        timeline,
        None,
        start,
        end,
    )
    return item


def _cue_alignment_ids(cue: Any) -> set[str]:
    if cue is None:
        return set()
    values: list[Any] = [
        getattr(cue, "segment_id", None),
        getattr(cue, "frame_id", None),
    ]
    values.extend(getattr(cue, "source_segment_ids", None) or [])
    values.extend(getattr(cue, "source_frame_ids", None) or [])
    metadata = getattr(cue, "metadata", None)
    if isinstance(metadata, dict):
        values.extend(_alignment_identity_values(metadata))
    return {str(value).strip() for value in values if str(value or "").strip()}


def _timeline_alignment_ids(item: dict[str, Any]) -> set[str]:
    return {
        str(value).strip()
        for value in _alignment_identity_values(item)
        if str(value or "").strip()
    }


def _alignment_identity_values(value: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in (
        "segment_id",
        "source_segment_id",
        "source_segment_ids",
        "frame_id",
        "source_frame_id",
        "source_frame_ids",
    ):
        current = value.get(key)
        if isinstance(current, list):
            values.extend(current)
        elif current is not None:
            values.append(current)
    return values


def _timeline_time_bounds(item: dict[str, Any]) -> tuple[float, float] | None:
    if "start" not in item and "start_seconds" not in item:
        return None
    start = _float(item.get("start", item.get("start_seconds")))
    end = _float(item.get("end", item.get("end_seconds", start)), start)
    if end < start:
        return None
    return start, end


def _timeline_intersects(item: dict[str, Any], start: float, end: float) -> bool:
    bounds = _timeline_time_bounds(item)
    if bounds is None:
        return False
    item_start, item_end = bounds
    # Positive-duration intervals need a positive overlap. Merely sharing an
    # endpoint is not evidence that a frame belongs to the spoken cue. Point
    # timestamps are handled explicitly by the bounded rounding path below.
    if end <= start or item_end <= item_start:
        return False
    return min(end, item_end) - max(start, item_start) > 0.0


def _timeline_rounding_tolerance_allowed(
    item: dict[str, Any], start: float, end: float
) -> bool:
    bounds = _timeline_time_bounds(item)
    if bounds is None:
        return False
    item_start, item_end = bounds
    return end <= start or item_end <= item_start


def _timeline_overlap_seconds(item: dict[str, Any], start: float, end: float) -> float:
    bounds = _timeline_time_bounds(item)
    if bounds is None:
        return 0.0
    item_start, item_end = bounds
    return max(0.0, min(end, item_end) - max(start, item_start))


def _timeline_distance_seconds(item: dict[str, Any], start: float, end: float) -> float:
    bounds = _timeline_time_bounds(item)
    if bounds is None:
        return float("inf")
    item_start, item_end = bounds
    if item_end < start:
        return start - item_end
    if end < item_start:
        return item_start - end
    return 0.0


def _timeline_time_sort_key(item: dict[str, Any], start: float, end: float) -> tuple[float, float, float, str]:
    bounds = _timeline_time_bounds(item)
    if bounds is None:
        return float("inf"), 0.0, float("inf"), str(item.get("index") or "")
    item_start, item_end = bounds
    cue_midpoint = (start + end) / 2.0
    item_midpoint = (item_start + item_end) / 2.0
    return (
        _timeline_distance_seconds(item, start, end),
        -_timeline_overlap_seconds(item, start, end),
        abs(item_midpoint - cue_midpoint),
        str(item.get("index") or ""),
    )


def _timeline_alignment_record(
    item: dict[str, Any],
    start: float,
    end: float,
    *,
    basis: str,
    window_seconds: float,
    shared_ids: list[str] | None = None,
) -> dict[str, Any]:
    bounds = _timeline_time_bounds(item)
    item_start, item_end = bounds if bounds is not None else (None, None)
    result = {
        "status": "matched",
        "basis": basis,
        "cue_start": start,
        "cue_end": end,
        "timeline_index": item.get("index"),
        "timeline_start": item_start,
        "timeline_end": item_end,
        "overlap_seconds": _timeline_overlap_seconds(item, start, end),
        "distance_seconds": _timeline_distance_seconds(item, start, end),
        "max_window_seconds": max(0.0, window_seconds),
    }
    if shared_ids:
        result["shared_ids"] = list(shared_ids)
    return result


def _load_sidecar_sources(root: Path, manifest: dict[str, Any], timeline: list[Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Any, str]] = []
    for key, source_type in SIDE_SOURCE_MANIFEST_KEYS:
        value = manifest.get(key)
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values, start=1):
            if item:
                source_id = key if len(values) == 1 else f"{key}_{index}"
                candidates.append((source_id, item, source_type))
    for name, source_type in SIDE_SOURCE_ROOT_FILES:
        candidates.append((name, name, source_type))
    sources: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source_id, value, source_type in candidates:
        path = _bundle_path(root, value)
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        try:
            cues = parse_transcript(path)
        except Exception:
            continue
        if cues:
            seen.add(resolved)
            metadata = _sidecar_source_metadata(path)
            sources.append({"source_id": source_id, "source_type": source_type, "path": str(path), "cues": cues, **metadata})
    timeline_cues = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        text = _flatten_text(item.get("subtitle") or item.get("caption") or item.get("original_subtitle"))
        if not text:
            continue
        start = _float(item.get("start", item.get("start_seconds", 0.0)))
        end = _float(item.get("end", item.get("end_seconds", start)), start)
        timeline_cues.append(type("Cue", (), {"start": start, "end": end, "text": text})())
    if timeline_cues:
        sources.append({"source_id": "timeline_subtitle", "source_type": "platform_subtitle", "path": str(root / "timeline.json"), "cues": timeline_cues})
    return sources


def _sidecar_source_metadata(path: Path) -> dict[str, Any]:
    payload = _read_optional_json(path) if path.suffix.lower() == ".json" else {}
    provider_value = payload.get("provider") if isinstance(payload, dict) else ""
    model_value = payload.get("model") if isinstance(payload, dict) else ""
    if isinstance(provider_value, list):
        provider = ",".join(sorted({str(item) for item in provider_value if str(item)}))
    else:
        provider = str(provider_value or "")
    if isinstance(model_value, list):
        model = ",".join(sorted({str(item) for item in model_value if str(item)}))
    else:
        model = str(model_value or "")
    timing_source = str(payload.get("timing_source") or "") if isinstance(payload, dict) else ""
    return {
        "provider": provider,
        "model": model,
        "artifact_sha256": _sha256_file(path),
        "timing_inferred": bool(timing_source and "inferred" in timing_source.lower()),
    }


def _metadata_evidence(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[str] = []
    for key in METADATA_TEXT_KEYS:
        value = manifest.get(key)
        if value:
            parts.append(f"{key}: {_flatten_text(value)}")
    for key in ("source_metadata", "page_metadata", "webpage_metadata", "vdo_handoff"):
        value = manifest.get(key)
        text = _flatten_text(value)
        if text:
            parts.append(f"{key}: {text}")
    metadata_paths = [root / name for name in ("source-metadata.json", "page-metadata.json", "webpage-metadata.json", "vdo-handoff.json")]
    pointer = str(manifest.get("page_metadata_json") or "").strip()
    if pointer:
        metadata_paths.insert(0, root / pointer)
    metadata_paths.insert(0, root / "source" / "page-metadata.json")
    seen_paths: set[Path] = set()
    for path in metadata_paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.exists():
            continue
        data = _read_optional_json(path)
        text = _flatten_text(data)
        if text:
            parts.append(f"{path.name}: {text}")
    text = " ".join(part for part in parts if part).strip()
    if not text:
        return []
    return [{"evidence_id": "metadata_manifest", "source_type": "page_metadata", "path": str(root / "manifest.json"), "text": text[:2000]}]


def _sidecar_evidence_for_cue(cue: Any, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = _float(getattr(cue, "start", 0.0))
    end = _float(getattr(cue, "end", start), start)
    raw = str(getattr(cue, "text", "") or "").strip()
    rows: list[dict[str, Any]] = []
    for source in sources:
        matches = _overlap_cues(source.get("cues") or [], start, end)
        if not matches:
            continue
        texts: list[str] = []
        seen_text: set[str] = set()
        for match in matches:
            text = str(getattr(match, "text", "") or "").strip()
            key = _normalize_compact(text)
            if not text or not key or key in seen_text:
                continue
            seen_text.add(key)
            texts.append(text)
        text = " ".join(texts).strip()
        if not text or _normalize_compact(text) == _normalize_compact(raw):
            continue
        source_id = str(source.get("source_id") or "subtitle")
        rows.append({
            "evidence_id": f"{source_id}_{len(rows)+1}",
            "source_type": str(source.get("source_type") or "platform_subtitle"),
            "start": min(_float(getattr(match, "start", start)) for match in matches),
            "end": max(_float(getattr(match, "end", end)) for match in matches),
            "path": source.get("path"),
            "provider": source.get("provider"),
            "model": source.get("model"),
            "artifact_sha256": source.get("artifact_sha256"),
            "timing_inferred": bool(source.get("timing_inferred")),
            "text": text[:1200],
        })
    return rows


def _overlap_cues(cues: list[Any], start: float, end: float, *, limit: int = 32) -> list[Any]:
    overlapping: list[tuple[float, float, Any]] = []
    for cue in cues:
        cue_start = _float(getattr(cue, "start", 0.0))
        cue_end = _float(getattr(cue, "end", cue_start), cue_start)
        overlap = min(end, cue_end) - max(start, cue_start)
        if overlap > 0:
            overlapping.append((cue_start, cue_end, cue))
    if overlapping:
        overlapping.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in overlapping[: max(1, int(limit or 1))]]
    nearest = _best_overlap_cue(cues, start, end)
    return [nearest] if nearest is not None else []


def _best_overlap_cue(cues: list[Any], start: float, end: float) -> Any | None:
    best: tuple[float, Any] | None = None
    for cue in cues:
        cue_start = _float(getattr(cue, "start", 0.0))
        cue_end = _float(getattr(cue, "end", cue_start), cue_start)
        overlap = min(end, cue_end) - max(start, cue_start)
        if overlap < 0:
            continue
        score = overlap if overlap > 0 else 0.01 / (1.0 + abs(((cue_start + cue_end) / 2) - ((start + end) / 2)))
        if best is None or score > best[0]:
            best = (score, cue)
    return best[1] if best else None


def _evidence_for_cue(
    cue: Any,
    timeline_item: dict[str, Any] | None,
    *,
    segment_index: int = 0,
    timeline_alignment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    asr_evidence = {
        "evidence_id": f"asr_segment_{segment_index}",
        "source_type": "asr_or_subtitle",
        "start": _float(getattr(cue, "start", 0.0)),
        "end": _float(getattr(cue, "end", 0.0)),
        "text": str(getattr(cue, "text", "") or ""),
    }
    if timeline_alignment:
        asr_evidence["timeline_alignment"] = dict(timeline_alignment)
    evidence = [asr_evidence]
    from .evidence_lineage import project_recognition_provenance
    asr_evidence['recognition_provenance'] = project_recognition_provenance({},
        selected_field=['asr_cue'], text=asr_evidence['text'])
    if isinstance(timeline_item, dict):
        index = timeline_item.get("index", "")
        for key, source_type in [("visual_text", "ocr"), ("structured_visual", "structured_visual")]:
            selector = visual_text_source_selector(timeline_item) if key == 'visual_text' else [key]
            # Structured evidence has its own row; do not duplicate it as OCR.
            if key == 'visual_text' and selector == ['structured_visual']:
                continue
            text = _flatten_text(timeline_item.get(selector[0])) if selector else ''
            if text:
                row = {
                    "evidence_id": f"timeline_{index}_{key}",
                    "source_type": (
                        "background_ocr" if key == "visual_text" and timeline_item.get("visual_text_role") == "incidental_background"
                        else source_type
                    ),
                    "timeline_index": index,
                    "evidence_level": _timeline_direct_evidence_level(
                        timeline_item, key=key
                    ),
                    "text": text[:1200],
                    "recognition_provenance": visual_recognition_provenance(timeline_item, selector, text=text),
                }
                if timeline_alignment:
                    row["timeline_alignment"] = dict(timeline_alignment)
                evidence.append(row)
        for key, source_type, state, content_fields in (
            (
                "visual_understanding",
                "visual_understanding",
                visual_evidence_state(timeline_item),
                VISUAL_CONTENT_FIELDS,
            ),
            (
                "temporal_visual_understanding",
                "temporal_visual",
                temporal_evidence_state(timeline_item),
                TEMPORAL_CONTENT_FIELDS,
            ),
        ):
            row = _qualified_timeline_visual_evidence(
                key=key,
                source_type=source_type,
                state=state,
                timeline_index=index,
                timeline_alignment=timeline_alignment,
                content_fields=content_fields,
            )
            if row:
                selector = understanding_source_selector(timeline_item, temporal=source_type == 'temporal_visual')
                # Use the same text projection as Summary so identity does not
                # depend on the correction pack's display-only 1200-char clip.
                from .smart_summary_input_pack import _qualified_evidence_text
                row['recognition_provenance'] = visual_recognition_provenance(
                    timeline_item, selector, text=_qualified_evidence_text(state))
                evidence.append(row)
        tag_parts = []
        # Keep actual tagger observations separate from their UI text copies.
        # A list position is a locator, never an independent-source identity.
        for prefix, owner in (([], timeline_item), (['integrated_visual'], timeline_item.get('integrated_visual'))):
            if not isinstance(owner, dict):
                continue
            annotations = owner.get('tagger_annotations')
            if not isinstance(annotations, list):
                continue
            for position, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                text = _flatten_text(annotation)
                if not text:
                    continue
                selector = [*prefix, 'tagger_annotations', position]
                row = {'evidence_id': f"timeline_{index}_{'_'.join(str(part) for part in selector)}",
                       'source_type': 'tagger', 'timeline_index': index,
                       'evidence_level': 'candidate_weak', 'text': text[:1200],
                       'recognition_provenance': visual_recognition_provenance(timeline_item, selector, text=text)}
                if timeline_alignment:
                    row['timeline_alignment'] = dict(timeline_alignment)
                evidence.append(row)
        for key in TAGGER_KEYS:
            if key == 'tagger_annotations' or (timeline_item.get('tagger_annotations') and key in {'tagger_tags','tagger_visual_summary','tagger_time_axis'}):
                continue
            value = timeline_item.get(key)
            text = _flatten_text(value)
            if text:
                tag_parts.append(f"{key}: {text}")
        integrated = timeline_item.get("integrated_visual") if isinstance(timeline_item.get("integrated_visual"), dict) else {}
        for key in TAGGER_KEYS:
            if key == 'tagger_annotations' or (integrated.get('tagger_annotations') and key in {'tagger_tags','tagger_visual_summary','tagger_time_axis'}):
                continue
            text = _flatten_text(integrated.get(key))
            if text:
                tag_parts.append(f"integrated_visual.{key}: {text}")
        if tag_parts:
            row = {
                "evidence_id": f"timeline_{index}_tagger",
                "source_type": "tagger",
                "timeline_index": index,
                "evidence_level": "candidate_weak",
                "text": " | ".join(tag_parts)[:1200],
            }
            if timeline_alignment:
                row["timeline_alignment"] = dict(timeline_alignment)
            evidence.append(row)
    return evidence


def _timeline_direct_evidence_level(item: dict[str, Any], *, key: str) -> str:
    if key == "visual_text":
        if str(item.get("human_corrected_visual_text") or "").strip():
            return "confirmed"
        level_keys = ("ocr_evidence_level", "visual_text_evidence_level")
        validation = item.get("ocr_validation")
    else:
        level_keys = ("structured_visual_evidence_level",)
        value = item.get("structured_visual")
        if isinstance(value, dict):
            explicit = str(value.get("evidence_level") or "").strip().lower()
            if explicit:
                return explicit
            validation = value.get("validation")
        else:
            validation = item.get("structured_visual_validation")
    for level_key in level_keys:
        explicit = str(item.get(level_key) or "").strip().lower()
        if explicit:
            return explicit
    if isinstance(validation, dict) and (
        validation.get("ok") is True
        or str(validation.get("status") or "").strip().lower()
        in {"ok", "passed", "confirmed"}
    ):
        return "confirmed"
    return "unverified"


def _qualified_timeline_visual_evidence(
    *,
    key: str,
    source_type: str,
    state: dict[str, Any],
    timeline_index: Any,
    timeline_alignment: dict[str, Any] | None,
    content_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    """Convert only consumable visual state into candidate-only evidence text."""

    if not state.get("consumable"):
        return None
    understanding = state.get("understanding")
    if not isinstance(understanding, dict):
        return None
    content = {
        field: understanding.get(field)
        for field in (*content_fields, "summary", "text", "description")
        if understanding.get(field) not in (None, "", [], {})
    }
    text = _strip_technical_visual_artifacts(_visual_mapping_text(content))
    if not text:
        return None
    evidence_level = str(state.get("evidence_level") or "confirmed")
    inference_basis = ""
    if evidence_level == "inferred":
        inference_basis = _strip_technical_visual_artifacts(
            _visual_mapping_text(state.get("inference_basis"))
        )
        suffix = f"；推断依据：{inference_basis}" if inference_basis else ""
        text = f"推断：画面可能显示{text}{suffix}"
    row = {
        "evidence_id": f"timeline_{timeline_index}_{key}",
        "source_type": source_type,
        "timeline_index": timeline_index,
        "evidence_level": evidence_level,
        "visual_evidence_source": str(state.get("source") or ""),
        "text": text[:1200],
        # This is evidence for candidate generation only.  It never authorizes
        # automatic transcript mutation or bypasses the existing validator.
        "candidate_evidence_only": True,
    }
    if inference_basis:
        row["inference_basis"] = inference_basis[:600]
    if timeline_alignment:
        row["timeline_alignment"] = dict(timeline_alignment)
    return row


def _validate_decision(
    decision: Any,
    candidates: dict[str, dict[str, Any]],
    *,
    idx: int,
    min_confidence: float,
    strict_model_contract: bool = False,
) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {"index": idx, "accepted": False, "reject_reasons": ["decision_not_object"]}
    candidate_id = str(decision.get("candidate_id") or "").strip()
    candidate = candidates.get(candidate_id, {})
    provided_original = decision.get("original_text")
    candidate_original = str(candidate.get("original_text") or "")
    original = str(provided_original or candidate_original).strip()
    corrected = str(decision.get("corrected_text") or decision.get("canonical") or decision.get("canonical_text") or "").strip()
    split_segments = _normalise_decision_segments(decision)
    merge_segment_indexes = _normalise_merge_segment_indexes(decision)
    if not corrected and split_segments:
        corrected = " ".join(str(row.get("text") or "").strip() for row in split_segments if str(row.get("text") or "").strip())
    kind = str(decision.get("correction_type") or candidate.get("correction_type") or "ordinary_word").strip()
    confidence = _float(decision.get("confidence"), 0.0)
    rationale = str(decision.get("semantic_rationale") or decision.get("rationale") or decision.get("reason") or "").strip()
    evidence_ids = [str(item) for item in (decision.get("evidence_ids") or []) if str(item).strip()]
    human_confirmed = bool(decision.get("human_confirmed") or decision.get("reviewed_by_human"))
    action = str(decision.get("action") or "").strip().lower()
    if not action:
        action = "replace" if bool(decision.get("accept", True)) else "reject"
    safe_to_apply = bool(decision.get("safe_to_apply", action == "replace"))
    needs_human_review = bool(decision.get("needs_human_review", False))
    no_change_candidate = bool(action == "reject" and corrected and original and _normalize_compact(corrected) == _normalize_compact(original))
    errors: list[str] = []
    if strict_model_contract:
        for key in STRICT_DECISION_REQUIRED_KEYS:
            if key not in decision:
                errors.append(f"missing_required_{key}")
    if (
        candidate
        and provided_original is not None
        and str(provided_original) != candidate_original
        and (
            strict_model_contract
            or _normalize_compact(str(provided_original)) not in _normalize_compact(candidate_original)
            and _normalize_compact(candidate_original) not in _normalize_compact(str(provided_original))
        )
    ):
        errors.append("original_text_mismatch")
    if strict_model_contract and "accept" in decision:
        expected_accept = action == "replace"
        if not isinstance(decision.get("accept"), bool) or decision.get("accept") != expected_accept:
            errors.append("accept_action_mismatch")
    if action not in {"replace", "keep_original", "needs_human_review", "review", "reject"}:
        errors.append("invalid_action")
    if action == "replace" and not safe_to_apply:
        errors.append("decision_not_accepted")
    elif action == "keep_original":
        if not human_confirmed:
            errors.append("keep_original_requires_human_confirmation")
    elif action != "replace" and not no_change_candidate:
        errors.append("decision_not_accepted")
    if needs_human_review and not human_confirmed:
        errors.append("needs_human_review")
    if not candidate_id:
        errors.append("missing_candidate_id")
    elif candidate_id not in candidates:
        errors.append("unknown_candidate_id")
    if kind not in VALID_TYPES:
        errors.append("invalid_correction_type")
    if not original:
        errors.append("missing_original_text")
    segment_errors = _validate_decision_segments(split_segments, candidate, kind=kind, human_confirmed=human_confirmed)
    merge_errors = _validate_merge_segment_indexes(merge_segment_indexes, candidate, kind=kind, human_confirmed=human_confirmed)
    errors.extend(segment_errors)
    errors.extend(merge_errors)
    if not corrected and action == "replace" and not split_segments:
        errors.append("missing_corrected_text")
    if corrected == original and action == "replace" and not split_segments:
        errors.append("unchanged_text")
    if confidence < min_confidence:
        errors.append("confidence_below_threshold")
    if not rationale:
        errors.append("missing_semantic_rationale")
    if not evidence_ids:
        errors.append("missing_evidence_ids")
    unknown_evidence = _unknown_evidence_ids(candidate, evidence_ids)
    if unknown_evidence:
        errors.append("unknown_evidence_ids")
    # Recompute from admitted pack evidence, never trust a model-provided vote
    # summary or let it hide counter-evidence by omitting citation IDs.
    lineage_support = _source_support_summary(original, corrected, candidate.get("evidence") or [])
    lineage_check = lineage_support["lineage"]
    subtitle_only_corroboration = bool(lineage_support["supports_candidate"]) and set(
        lineage_support["supports_candidate"]
    ) <= {"platform_subtitle", "embedded_subtitle"}
    # Primary ASR disagreement is the correction target, not by itself a veto:
    # retain the established dual-recognizer/lexicon recognition checks below.
    # Recognition corroboration does not establish independent factual sources.
    if action == "replace" and not human_confirmed and (
        lineage_check["intra_group_direct_conflict_count"]
        or lineage_check["unresolved_group_count"]
        or (subtitle_only_corroboration and lineage_check["intra_group_conflict_count"])
    ):
        errors.append("correlated_or_unresolved_evidence_requires_review")
    original_fact_values = _fact_value_markers(original)
    corrected_fact_values = _fact_value_markers(corrected)
    original_action_values = _action_markers(original)
    corrected_action_values = _action_markers(corrected)
    high_risk_fact_change = _is_high_risk_fact_change(kind=kind, original=original, corrected=corrected)
    high_risk_action_change = _is_high_risk_action_change(kind=kind, original=original, corrected=corrected)
    original_semantic_anchors = _semantic_anchor_map(original, declared_kind=kind)
    corrected_semantic_anchors = _semantic_anchor_map(corrected, declared_kind=kind)
    semantic_anchor_change_types = _semantic_anchor_change_types(
        original_semantic_anchors,
        corrected_semantic_anchors,
        declared_kind=kind,
        original=original,
        corrected=corrected,
    )
    high_risk_semantic_anchor_change = bool(semantic_anchor_change_types)
    semantic_anchor_support_source_types = _semantic_anchor_support_source_types(
        candidate,
        evidence_ids,
        corrected_text=corrected,
        corrected_anchors=corrected_semantic_anchors,
        changed_types=semantic_anchor_change_types,
    )
    linguistic_support_sources = _linguistic_support_sources(
        candidate,
        evidence_ids,
        corrected_text=corrected,
    )
    semantic_corroboration = _direct_corroboration_lineage(
        candidate, evidence_ids, corrected_text=corrected,
        corrected_anchors=corrected_semantic_anchors, changed_types=semantic_anchor_change_types,
    ) if high_risk_semantic_anchor_change else None
    linguistic_corroboration = _direct_corroboration_lineage(
        candidate, evidence_ids, corrected_text=corrected,
    ) if kind in {"ordinary_homophone", "collocation_anomaly"} else None
    if (
        action == "replace"
        and kind in {"ordinary_homophone", "collocation_anomaly"}
        and not human_confirmed
        and (len(linguistic_support_sources) < 2
             or linguistic_corroboration["declared_support_group_count"] < 2)
    ):
        errors.append(
            "linguistic_candidate_requires_human_or_two_independent_direct_evidence"
        )
    if (
        action == "replace"
        and high_risk_semantic_anchor_change
        and not human_confirmed
        and (len(semantic_anchor_support_source_types) < 2
             or semantic_corroboration["declared_support_group_count"] < 2)
    ):
        errors.append("semantic_anchor_change_requires_human_or_two_independent_direct_evidence")
    if action == "replace" and high_risk_fact_change and not human_confirmed and (confidence < max(0.95, min_confidence) or not _has_strong_number_evidence(candidate, evidence_ids, corrected_text=corrected)):
        errors.append("unsafe_fact_value_without_strong_evidence")
        errors.append("fact_value_requires_stronger_evidence_or_human_confirmation")
        if kind == "number":
            errors.append("unsafe_number_without_strong_evidence")
            errors.append("number_requires_stronger_evidence_or_human_confirmation")
    if action == "replace" and high_risk_action_change and not human_confirmed and (confidence < max(0.92, min_confidence) or not _has_strong_action_evidence(candidate, evidence_ids, corrected_text=corrected)):
        errors.append("unsafe_action_without_visual_or_human_evidence")
        errors.append("action_change_requires_visual_temporal_or_human_confirmation")
    if candidate.get("has_conflict") and not human_confirmed:
        conflict_threshold = max(0.92, min_confidence)
        if (
            not high_risk_fact_change
            and not high_risk_action_change
            and kind in {"proper_noun", "term", "ordinary_word", "ordinary_homophone", "collocation_anomaly", "concept"}
            and _has_candidate_discovery_evidence(candidate, evidence_ids)
        ):
            conflict_threshold = max(0.85, min_confidence)
        if confidence < conflict_threshold:
            errors.append("conflict_not_marked_for_review")
    if (
        action == "replace"
        and candidate.get("has_conflict")
        and not human_confirmed
        and kind in {"proper_noun", "term", "ordinary_word", "ordinary_homophone", "collocation_anomaly", "concept"}
        and any(str(item.get("source_type") or "") == "secondary_asr" for item in candidate.get("evidence", []) if isinstance(item, dict))
        and not _has_two_independent_secondary_asr_evidence(candidate, evidence_ids, corrected_text=corrected)
        and not _has_domain_lexicon_plus_secondary_asr_support(candidate, evidence_ids, corrected_text=corrected)
        and not _has_direct_non_asr_support(candidate, evidence_ids, corrected_text=corrected)
    ):
        errors.append("insufficient_independent_evidence_for_asr_conflict")
    timeline_indexes = decision.get("timeline_indexes") or candidate.get("timeline_indexes") or []
    arbitrated_no_change = bool(no_change_candidate and not errors)
    return {
        "candidate_id": candidate_id,
        "action": action,
        "segment_index": candidate.get("segment_index"),
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "time_range": candidate.get("time_range", ""),
        "timeline_indexes": timeline_indexes,
        "correction_type": kind,
        "original_text": original,
        "corrected_text": corrected,
        "segments": split_segments,
        "merge_segment_indexes": merge_segment_indexes,
        "confidence": confidence,
        "semantic_rationale": rationale,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "human_confirmed": human_confirmed,
        "safe_to_apply": safe_to_apply,
        "needs_human_review": needs_human_review,
        "high_risk_fact_change": high_risk_fact_change,
        "original_fact_values": original_fact_values,
        "corrected_fact_values": corrected_fact_values,
        "high_risk_action_change": high_risk_action_change,
        "original_action_values": original_action_values,
        "corrected_action_values": corrected_action_values,
        "high_risk_semantic_anchor_change": high_risk_semantic_anchor_change,
        "semantic_anchor_change_types": semantic_anchor_change_types,
        "original_semantic_anchors": {
            key: original_semantic_anchors.get(key, [])
            for key in semantic_anchor_change_types
        },
        "corrected_semantic_anchors": {
            key: corrected_semantic_anchors.get(key, [])
            for key in semantic_anchor_change_types
        },
        "semantic_anchor_support_source_types": semantic_anchor_support_source_types,
        "linguistic_support_sources": linguistic_support_sources,
        "semantic_anchor_corroboration": semantic_corroboration,
        "linguistic_corroboration": linguistic_corroboration,
        "evidence_lineage": lineage_check,
        "apply_scope": decision.get("apply_scope") or "segment",
        "accepted": bool(not errors and not arbitrated_no_change),
        "arbitrated_no_change": arbitrated_no_change,
        "semantic_decision_status": "arbitrated_no_change" if arbitrated_no_change else ("accepted" if not errors else "rejected"),
        "reject_reasons": errors,
        "candidate": candidate,
    }


def _semantic_anchor_map(text: str, *, declared_kind: str = "") -> dict[str, list[str]]:
    value = str(text or "").strip()
    patterns = {
        "negation": NEGATION_ANCHOR_RE,
        "condition": CONDITION_ANCHOR_RE,
        "exception": EXCEPTION_ANCHOR_RE,
        "person_name": PERSON_NAME_ANCHOR_RE,
        "organization": ORGANIZATION_ANCHOR_RE,
        "date_time": DATE_TIME_ANCHOR_RE,
        "money": MONEY_ANCHOR_RE,
        "percentage": PERCENTAGE_ANCHOR_RE,
    }
    result: dict[str, list[str]] = {}
    for anchor_type, pattern in patterns.items():
        matches = _normalise_semantic_anchor_values(pattern.findall(value))
        if matches:
            result[anchor_type] = matches
    if declared_kind in {"person_name", "organization"} and value:
        result.setdefault(declared_kind, [value])
    if declared_kind in SEMANTIC_ANCHOR_TYPES:
        result.setdefault(declared_kind, [])
    return result


def _normalise_semantic_anchor_values(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, tuple):
            value = next((item for item in value if str(item or "").strip()), "")
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _semantic_anchor_change_types(
    original_anchors: dict[str, list[str]],
    corrected_anchors: dict[str, list[str]],
    *,
    declared_kind: str,
    original: str,
    corrected: str,
) -> list[str]:
    changed = [
        anchor_type
        for anchor_type in sorted(set(original_anchors) | set(corrected_anchors))
        if [_normalize_compact(item) for item in original_anchors.get(anchor_type, [])]
        != [_normalize_compact(item) for item in corrected_anchors.get(anchor_type, [])]
    ]
    if (
        declared_kind in SEMANTIC_ANCHOR_TYPES
        and _normalize_compact(original) != _normalize_compact(corrected)
        and declared_kind not in changed
    ):
        changed.append(declared_kind)
    return sorted(changed)


def _semantic_anchor_support_source_types(
    candidate: dict[str, Any],
    evidence_ids: list[str],
    *,
    corrected_text: str,
    corrected_anchors: dict[str, list[str]],
    changed_types: list[str],
) -> list[str]:
    selected_ids = set(evidence_ids)
    support_types: set[str] = set()
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id") or "") not in selected_ids:
            continue
        source_type = str(item.get("source_type") or "")
        if source_type not in DIRECT_SEMANTIC_EVIDENCE_TYPES:
            continue
        evidence_level = str(item.get("evidence_level") or "").strip().lower()
        if evidence_level in {"inferred", "unavailable", "not_seen", "candidate_weak"}:
            continue
        evidence_text = _normalize_compact(str(item.get("text") or ""))
        if not evidence_text:
            continue
        supports_all = True
        for anchor_type in changed_types:
            anchors = [
                _normalize_compact(value)
                for value in corrected_anchors.get(anchor_type, [])
                if _normalize_compact(value)
            ]
            if anchors:
                if not any(anchor in evidence_text for anchor in anchors):
                    supports_all = False
                    break
            elif _normalize_compact(corrected_text) not in evidence_text:
                supports_all = False
                break
        if supports_all:
            support_types.add(source_type)
    return sorted(support_types)


def _linguistic_support_sources(
    candidate: dict[str, Any],
    evidence_ids: list[str],
    *,
    corrected_text: str,
) -> list[str]:
    """Return distinct direct sources that literally support a wording change."""

    selected_ids = set(evidence_ids)
    corrected = _normalize_compact(corrected_text)
    if not corrected:
        return []
    sources: set[str] = set()
    for item in candidate.get("evidence", []):
        if (
            not isinstance(item, dict)
            or str(item.get("evidence_id") or "") not in selected_ids
        ):
            continue
        source_type = str(item.get("source_type") or "")
        if source_type not in DIRECT_SEMANTIC_EVIDENCE_TYPES:
            continue
        if not _strong_direct_evidence_is_eligible(item):
            continue
        if corrected not in _normalize_compact(str(item.get("text") or "")):
            continue
        if source_type == "secondary_asr":
            identity = str(
                item.get("artifact_sha256")
                or item.get("provider")
                or item.get("path")
                or item.get("evidence_id")
                or ""
            ).strip()
            if identity:
                sources.add(f"secondary_asr:{identity}")
        else:
            sources.add(source_type)
    return sorted(sources)


def _direct_corroboration_lineage(candidate, evidence_ids, *, corrected_text,
                                  corrected_anchors=None, changed_types=None):
    """Apply existing source grouping at correction admission, not just reports.

    Source-type counts remain diagnostics/legacy gates, never an independence
    proof. Reuse existing literal support selectors and strong-evidence policy.
    Include neutral/unselected rows when grouping so parent aliases cannot be
    hidden by a model's citation selection. Declared roots are not verified
    factual independence; this adds a veto and does not relax any old gate.
    """
    from .evidence_lineage import aggregate_source_votes

    rows = candidate.get("evidence") or []
    selected = set(evidence_ids)
    votes = []
    for row in rows:
        name = str(row.get("evidence_id") or "") if isinstance(row, dict) else ""
        support = False
        if name in selected and _strong_direct_evidence_is_eligible(row):
            scoped = {"evidence": [row]}
            if changed_types is None:
                support = bool(_linguistic_support_sources(scoped, [name], corrected_text=corrected_text))
            else:
                support = bool(_semantic_anchor_support_source_types(
                    scoped, [name], corrected_text=corrected_text,
                    corrected_anchors=corrected_anchors, changed_types=changed_types))
        votes.append({"vote": "supports_candidate" if support else "neutral",
                      "source_weight": 1 if support else 0})
    lineage = aggregate_source_votes(rows, votes)
    count = sum(group["observation_identity_status"] == "declared_root"
                and group["weights"]["candidate_weight"] > 0 and not group["problems"]
                for group in lineage["groups"])
    return {"declared_support_group_count": count, "lineage": lineage,
            "scope": "literal_correction_corroboration_not_independent_factual_truth"}


def _normalise_decision_segments(decision: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = decision.get("segments") or decision.get("split_segments") or decision.get("replacement_segments") or []
    if not isinstance(raw_segments, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_segments):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("corrected_text") or "").strip()
        rows.append(
            {
                "index": idx,
                "start": _float(row.get("start", row.get("start_seconds", 0.0))),
                "end": _float(row.get("end", row.get("end_seconds", row.get("start", 0.0)))) ,
                "text": text,
            }
        )
    return rows


def _validate_decision_segments(segments: list[dict[str, Any]], candidate: dict[str, Any], *, kind: str, human_confirmed: bool) -> list[str]:
    if not segments:
        return []
    errors: list[str] = []
    if kind not in {"segment_boundary", "punctuation"}:
        errors.append("segments_only_allowed_for_boundary_or_punctuation")
    if len(segments) < 2:
        errors.append("split_segments_require_at_least_two_segments")
    if not human_confirmed:
        errors.append("split_segments_require_human_confirmation")
    candidate_start = _float(candidate.get("start", 0.0))
    candidate_end = _float(candidate.get("end", candidate_start), candidate_start)
    previous_end: float | None = None
    for row in segments:
        start = _float(row.get("start"), candidate_start)
        end = _float(row.get("end"), start)
        text = str(row.get("text") or "").strip()
        if not text:
            errors.append("split_segment_missing_text")
        if end <= start:
            errors.append("split_segment_invalid_time_range")
        if previous_end is not None and start < previous_end - 0.25:
            errors.append("split_segments_not_time_ordered")
        if candidate_end > candidate_start and (start < candidate_start - 1.0 or end > candidate_end + 1.0):
            errors.append("split_segment_outside_candidate_range")
        previous_end = end
    return sorted(set(errors))

def _normalise_merge_segment_indexes(decision: dict[str, Any]) -> list[int]:
    values = decision.get("merge_segment_indexes") or decision.get("source_segment_indexes") or decision.get("merge_segments") or []
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            index = int(value)
        except Exception:
            continue
        if index not in result:
            result.append(index)
    return result


def _validate_merge_segment_indexes(indexes: list[int], candidate: dict[str, Any], *, kind: str, human_confirmed: bool) -> list[str]:
    if not indexes:
        return []
    errors: list[str] = []
    if kind not in {"segment_boundary", "punctuation"}:
        errors.append("merge_segments_only_allowed_for_boundary_or_punctuation")
    if len(indexes) < 2:
        errors.append("merge_segments_require_at_least_two_segments")
    if not human_confirmed:
        errors.append("merge_segments_require_human_confirmation")
    if any(index < 0 for index in indexes):
        errors.append("merge_segments_invalid_index")
    if indexes != sorted(indexes):
        errors.append("merge_segments_not_time_ordered")
    candidate_index = candidate.get("segment_index")
    if isinstance(candidate_index, int) and candidate_index not in indexes:
        errors.append("merge_segments_must_include_candidate_segment")
    if isinstance(candidate_index, int) and indexes and indexes[0] != candidate_index:
        errors.append("merge_segments_must_start_with_candidate_segment")
    return sorted(set(errors))

def _unknown_evidence_ids(candidate: dict[str, Any], evidence_ids: list[str]) -> list[str]:
    known = {str(item.get("evidence_id")) for item in candidate.get("evidence", []) if isinstance(item, dict)}
    return [item for item in evidence_ids if item not in known]



def _has_candidate_discovery_evidence(candidate: dict[str, Any], evidence_ids: list[str]) -> bool:
    selected_ids = set(evidence_ids)
    if not selected_ids:
        return False
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id")) not in selected_ids:
            continue
        if str(item.get("source_type") or "") == "candidate_discovery_suggestion":
            return True
    return False
def _has_two_independent_secondary_asr_evidence(
    candidate: dict[str, Any],
    evidence_ids: list[str],
    *,
    corrected_text: str,
) -> bool:
    selected_ids = set(evidence_ids)
    corrected = _normalize_compact(corrected_text)
    # Distinct files/runs are not distinct recognizers. This is only agreement
    # about the transcription, never proof of the speaker's factual claim.
    recognitions: set[tuple[str, str]] = set()
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id") or "") not in selected_ids:
            continue
        if str(item.get("source_type") or "") != "secondary_asr":
            continue
        evidence_text = _normalize_compact(str(item.get("text") or ""))
        if corrected and corrected not in evidence_text:
            continue
        backend = str(item.get("model_actual") or item.get("provider") or "").strip().lower()
        digest = str(item.get("artifact_sha256") or "").lower()
        if backend and re.fullmatch(r"[0-9a-f]{64}", digest) and not item.get("derived_from_evidence_ids"):
            recognitions.add((backend, digest))
    return any(a[0] != b[0] and a[1] != b[1] for a in recognitions for b in recognitions)


def _has_domain_lexicon_plus_secondary_asr_support(
    candidate: dict[str, Any],
    evidence_ids: list[str],
    *,
    corrected_text: str,
) -> bool:
    selected_ids = set(evidence_ids)
    corrected = _normalize_compact(corrected_text)
    lexicon_supported = False
    secondary_identities: set[str] = set()
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id") or "") not in selected_ids:
            continue
        source_type = str(item.get("source_type") or "")
        evidence_text = _normalize_compact(str(item.get("text") or ""))
        if corrected and corrected not in evidence_text:
            continue
        if source_type == "explicit_domain_lexicon":
            lexicon_supported = True
        elif source_type == "secondary_asr":
            identity = str(item.get("artifact_sha256") or item.get("provider") or item.get("path") or "").strip()
            if identity:
                secondary_identities.add(identity)
    return lexicon_supported and bool(secondary_identities)


def _has_direct_non_asr_support(
    candidate: dict[str, Any],
    evidence_ids: list[str],
    *,
    corrected_text: str,
) -> bool:
    selected_ids = set(evidence_ids)
    corrected = _normalize_compact(corrected_text)
    direct_types = {"human_note", "ocr", "structured_visual", "platform_subtitle", "embedded_subtitle"}
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id") or "") not in selected_ids:
            continue
        if str(item.get("source_type") or "") not in direct_types:
            continue
        if str(item.get("source_type") or "") == "human_note":
            return True
        if corrected and corrected in _normalize_compact(str(item.get("text") or "")):
            return True
    return False


def _has_strong_number_evidence(candidate: dict[str, Any], evidence_ids: list[str], *, corrected_text: str = "") -> bool:
    strong_types = {"ocr", "structured_visual", "platform_subtitle", "embedded_subtitle", "human_note"}
    selected_ids = set(evidence_ids)
    corrected_markers = set(_fact_value_markers(corrected_text))
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id")) not in selected_ids:
            continue
        if str(item.get("source_type")) not in strong_types:
            continue
        if not _strong_direct_evidence_is_eligible(item):
            continue
        evidence_text = str(item.get("text") or "")
        if not corrected_markers:
            return True
        evidence_markers = set(_fact_value_markers(evidence_text))
        if corrected_markers & evidence_markers:
            return True
        if corrected_text and corrected_text.strip() and corrected_text.strip().lower() in evidence_text.lower():
            return True
    return False


def _is_high_risk_fact_change(*, kind: str, original: str, corrected: str) -> bool:
    if kind == "number":
        return True
    if not original or not corrected or original == corrected:
        return False
    original_values = _fact_value_markers(original)
    corrected_values = _fact_value_markers(corrected)
    return bool(original_values or corrected_values) and original_values != corrected_values


def _fact_value_markers(text: str) -> list[str]:
    values = [item.strip().lower().replace(" ", "") for item in NUMBER_RE.findall(text or "") if item.strip()]
    values.extend(item.strip().lower().replace(" ", "") for item in CHINESE_FACT_VALUE_RE.findall(text or "") if item.strip())
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _is_high_risk_action_change(*, kind: str, original: str, corrected: str) -> bool:
    if not original or not corrected or original == corrected:
        return False
    original_actions = _action_markers(original)
    corrected_actions = _action_markers(corrected)
    if kind == "action":
        return bool(original_actions or corrected_actions) and original_actions != corrected_actions
    return bool(original_actions and corrected_actions and original_actions != corrected_actions)


def _action_markers(text: str) -> list[str]:
    result: list[str] = []
    for value in ACTION_VERB_RE.findall(text or ""):
        if value and value not in result:
            result.append(value)
    return result


def _has_strong_action_evidence(candidate: dict[str, Any], evidence_ids: list[str], *, corrected_text: str = "") -> bool:
    strong_types = {"visual_understanding", "temporal_visual", "structured_visual", "ocr", "human_note"}
    selected_ids = set(evidence_ids)
    corrected_actions = set(_action_markers(corrected_text))
    for item in candidate.get("evidence", []):
        if not isinstance(item, dict) or str(item.get("evidence_id")) not in selected_ids:
            continue
        if str(item.get("source_type")) not in strong_types:
            continue
        if not _strong_direct_evidence_is_eligible(item):
            continue
        if str(item.get("source_type")) == "human_note":
            return True
        evidence_text = str(item.get("text") or "")
        if not corrected_actions:
            return True
        evidence_actions = set(_action_markers(evidence_text))
        if corrected_actions & evidence_actions:
            return True
        if corrected_text and corrected_text.strip() and corrected_text.strip() in evidence_text:
            return True
    return False


def _strong_direct_evidence_is_eligible(item: dict[str, Any]) -> bool:
    """Keep weak/unconfirmed visual projections out of automatic correction.

    Platform and embedded subtitles retain their existing independently hashed
    sidecar contract. Visual/OCR evidence must explicitly be confirmed, while
    tagger output can never prove an action regardless of a caller-supplied
    label. Human notes remain the explicit operator-confirmed route.
    """

    source_type = str(item.get("source_type") or "").strip()
    if source_type == "human_note":
        return True
    if source_type == "tagger":
        return False
    if source_type in {
        "ocr",
        "structured_visual",
        "visual_understanding",
        "temporal_visual",
    }:
        level = str(item.get("evidence_level") or "unverified").strip().lower()
        return level == "confirmed" and item.get("consumable", True) is not False
    return source_type in {"platform_subtitle", "embedded_subtitle"}


def _review_note_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("reviews", "items", "notes", "decisions"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    return []


def _decision_from_review_row(row: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    requested_candidate_id = str(row.get("candidate_id") or row.get("semantic_candidate_id") or "").strip()
    row_original = str(row.get("original_text") or "").strip()
    candidate = candidates.get(requested_candidate_id)
    if row_original and (not candidate or str(candidate.get("original_text") or "").strip() != row_original):
        candidate = _unique_review_candidate(row, candidates, original_text=row_original)
    if not candidate:
        return None
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    rebound_from_candidate_id = requested_candidate_id if requested_candidate_id and requested_candidate_id != candidate_id else ""
    status = str(row.get("review_status") or row.get("status") or row.get("action") or "").strip().lower()
    corrected_text = str(row.get("corrected_text") or row.get("corrected_transcript") or row.get("corrected_semantic_text") or row.get("suggested_text") or "").strip()
    comment = str(row.get("review_note") or row.get("comment") or row.get("notes") or "human review confirmed semantic correction").strip()
    evidence_source = candidate.get("evidence_ids") if rebound_from_candidate_id else (row.get("evidence_ids") or candidate.get("evidence_ids"))
    evidence_ids = [str(item) for item in (evidence_source or []) if str(item).strip()]
    split_segments = _normalise_decision_segments(row)
    merge_segment_indexes = _normalise_merge_segment_indexes(row)
    if status in {"accept_correction", "accepted_correction", "edit_correction", "corrected_transcript", "corrected_semantic_correction", "replace"}:
        if not corrected_text and not split_segments and not merge_segment_indexes:
            return None
        decision = {
            "candidate_id": candidate_id,
            "action": "replace",
            "correction_type": row.get("correction_type") or candidate.get("correction_type") or "ordinary_word",
            "original_text": candidate.get("original_text") or "",
            "corrected_text": corrected_text,
            "confidence": _float(row.get("confidence"), 1.0),
            "semantic_rationale": comment,
            "evidence_ids": evidence_ids,
            "timeline_indexes": row.get("timeline_indexes") or candidate.get("timeline_indexes") or [],
            "safe_to_apply": True,
            "needs_human_review": False,
            "human_confirmed": True,
        }
        if rebound_from_candidate_id:
            decision["rebound_from_candidate_id"] = rebound_from_candidate_id
        if split_segments:
            decision["segments"] = split_segments
        if merge_segment_indexes:
            decision["merge_segment_indexes"] = merge_segment_indexes
        return decision
    if status in {"keep_original", "accepted_known_gap", "reject", "rejected"}:
        return {
            "candidate_id": candidate_id,
            "action": "keep_original",
            "correction_type": row.get("correction_type") or candidate.get("correction_type") or "ordinary_word",
            "original_text": candidate.get("original_text") or "",
            "corrected_text": candidate.get("original_text") or "",
            "confidence": _float(row.get("confidence"), 1.0),
            "semantic_rationale": comment,
            "evidence_ids": evidence_ids,
            "timeline_indexes": row.get("timeline_indexes") or candidate.get("timeline_indexes") or [],
            "safe_to_apply": False,
            "needs_human_review": False,
            "human_confirmed": True,
        }
    if status in {"needs_more_evidence", "needs_human_review", "needs_rerun_asr", "needs_rerun_ocr"}:
        return {
            "candidate_id": candidate_id,
            "action": "needs_human_review",
            "correction_type": row.get("correction_type") or candidate.get("correction_type") or "ordinary_word",
            "original_text": candidate.get("original_text") or "",
            "corrected_text": corrected_text or candidate.get("candidate_text") or candidate.get("suggested_text") or "",
            "confidence": _float(row.get("confidence"), 0.0),
            "semantic_rationale": comment,
            "evidence_ids": evidence_ids,
            "timeline_indexes": row.get("timeline_indexes") or candidate.get("timeline_indexes") or [],
            "safe_to_apply": False,
            "needs_human_review": True,
            "human_confirmed": False,
        }
    return None


def _unique_review_candidate(
    row: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    *,
    original_text: str,
) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in candidates.values()
        if str(candidate.get("original_text") or "").strip() == original_text
    ]
    correction_type = str(row.get("correction_type") or "").strip()
    if correction_type:
        typed = [candidate for candidate in matches if str(candidate.get("correction_type") or "") == correction_type]
        if typed:
            matches = typed
    segment_value = row.get("source_segment_index")
    if segment_value is None:
        segment_value = row.get("segment_index")
    if segment_value is not None:
        try:
            segment_index = int(segment_value)
        except (TypeError, ValueError):
            segment_index = None
        if segment_index is not None:
            segmented = [candidate for candidate in matches if candidate.get("segment_index") == segment_index]
            if segmented:
                matches = segmented
    return matches[0] if len(matches) == 1 else None


def _render_review_import_markdown(payload: dict[str, Any], skipped: list[dict[str, Any]]) -> str:
    lines = ["# 转写语义纠错人工复核导入", "", f"- Decisions: `{len(payload.get('decisions') or [])}`", f"- Skipped: `{len(skipped)}`", ""]
    for row in payload.get("decisions") or []:
        lines.append(f"- `{row.get('candidate_id')}` action=`{row.get('action')}` `{row.get('original_text')}` -> `{row.get('corrected_text')}`")
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        for row in skipped:
            lines.append(f"- row `{row.get('row_number')}` candidate=`{row.get('candidate_id')}` reason=`{row.get('reason')}`")
    return "\n".join(lines) + "\n"

def _semantic_review_rows(rejected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rejected:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        reasons = [str(item) for item in row.get("reject_reasons") or [] if str(item)]
        if row.get("action") in {"keep_original", "reject"} and reasons == ["decision_not_accepted"]:
            continue
        rows.append({
            "target_type": "transcript_semantic_correction",
            "candidate_id": row.get("candidate_id"),
            "segment_index": row.get("segment_index"),
            "timeline_indexes": row.get("timeline_indexes") or candidate.get("timeline_indexes") or [],
            "start": row.get("start"),
            "end": row.get("end"),
            "time_range": row.get("time_range") or candidate.get("time_range") or "",
            "correction_type": row.get("correction_type"),
            "original_text": row.get("original_text"),
            "suggested_text": row.get("corrected_text") or candidate.get("candidate_text") or candidate.get("suggested_text") or "",
            "confidence": row.get("confidence"),
            "semantic_rationale": row.get("semantic_rationale") or row.get("rationale") or "",
            "evidence_ids": row.get("evidence_ids") or [],
            "reject_reasons": reasons,
            "high_risk_fact_change": bool(row.get("high_risk_fact_change")),
            "original_fact_values": row.get("original_fact_values") or [],
            "corrected_fact_values": row.get("corrected_fact_values") or [],
            "high_risk_action_change": bool(row.get("high_risk_action_change")),
            "original_action_values": row.get("original_action_values") or [],
            "corrected_action_values": row.get("corrected_action_values") or [],
            "quality_issues": [f"semantic_correction_{reason}" for reason in reasons],
            "evidence_excerpt": _short_evidence_excerpt(candidate),
            "context_text": candidate.get("context_text") or "",
            "candidate": candidate,
            "suggested_status": "corrected_transcript" if row.get("corrected_text") else "needs_more_evidence",
            "suggested_action": "人工核对 ASR/字幕/OCR/视觉/网页证据；确认后重新导入 transcript-semantic-correction-result.codex.md，或保留原文。",
            "closed": False,
        })
    return rows


def _semantic_review_payload(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_correction_review.v1",
        "bundle_dir": str(root),
        "review_required_count": len(rows),
        "items": rows,
        "updated_at": now_iso(),
    }


def _status_detail_summary(pack: Any, validation: Any, root: Path) -> dict[str, Any]:
    candidates = [row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    candidate_groups = [row for row in (pack.get("candidate_groups") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    if candidates and not candidate_groups:
        candidate_groups = _assign_candidate_groups(candidates)
    candidates_by_id = {str(row.get("candidate_id") or ""): row for row in candidates if str(row.get("candidate_id") or "")}
    rejected = [row for row in (validation.get("rejected_decisions") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    no_change = [row for row in (validation.get("arbitrated_no_change_decisions") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    review_rows = [row for row in (validation.get("review_rows") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    if not review_rows:
        review_payload = _read_optional_json(root / "transcript-semantic-correction-review.json")
        review_rows = [row for row in (review_payload.get("items") if isinstance(review_payload, dict) else []) or [] if isinstance(row, dict)]
    chapters = _load_semantic_chapter_ranges(root)
    evidence_source_counts: dict[str, int] = {}
    for row in candidates:
        for source_type in row.get("evidence_source_types") or []:
            key = str(source_type or "unknown")
            evidence_source_counts[key] = evidence_source_counts.get(key, 0) + 1
    rejection_reason_counts: dict[str, int] = {}
    for row in rejected:
        for reason in row.get("reject_reasons") or []:
            key = str(reason or "unknown")
            rejection_reason_counts[key] = rejection_reason_counts.get(key, 0) + 1
    review_items = []
    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id, {})
        start = _float(row.get("start", candidate.get("start", 0.0)))
        end = _float(row.get("end", candidate.get("end", start)), start)
        chapter = _chapter_for_semantic_time(chapters, start, end)
        review_items.append(
            {
                "candidate_id": candidate_id,
                "correction_type": str(row.get("correction_type") or candidate.get("correction_type") or ""),
                "risk_level": str(row.get("risk_level") or candidate.get("risk_level") or "unknown"),
                "chapter_index": chapter.get("chapter_index"),
                "chapter_title": chapter.get("chapter_title", ""),
                "chapter_time_range": chapter.get("chapter_time_range", ""),
                "time_range": str(row.get("time_range") or candidate.get("time_range") or ""),
                "start": start,
                "end": end,
                "original_text": str(row.get("original_text") or candidate.get("original_text") or "")[:160],
                "suggested_text": str(row.get("suggested_text") or row.get("corrected_text") or candidate.get("suggested_text") or "")[:160],
                "reject_reasons": [str(item) for item in row.get("reject_reasons") or row.get("quality_issues") or [] if str(item)],
                "evidence_source_types": [str(item) for item in row.get("evidence_source_types") or candidate.get("evidence_source_types") or [] if str(item)],
                "source_support_summary": candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {},
            }
        )
    chapter_risk_summary = _semantic_chapter_risk_summary(candidates, review_items, chapters)
    semantic_attention_items = _semantic_attention_items(candidates, chapters)
    source_vote_summary = _source_vote_summary(candidates)
    return {
        "candidate_type_counts": _count_values(row.get("correction_type") for row in candidates),
        "risk_level_counts": _count_values(row.get("risk_level") for row in candidates),
        "candidate_group_count": len(candidate_groups),
        "candidate_group_preview": candidate_groups[:8],
        "evidence_source_counts": evidence_source_counts,
        "validation_rejection_reason_counts": rejection_reason_counts,
        "arbitrated_no_change_count": len(no_change),
        "review_required_items": review_items,
        "review_required_preview": review_items[:8],
        "semantic_attention_items": semantic_attention_items,
        "semantic_attention_preview": semantic_attention_items[:10],
        "source_vote_summary": source_vote_summary,
        "chapter_risk_summary": chapter_risk_summary,
    }




def _source_vote_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_dominant_side: dict[str, int] = {}
    by_candidate_support_source: dict[str, int] = {}
    by_original_support_source: dict[str, int] = {}
    conflict_rows: list[dict[str, Any]] = []
    candidate_weight_total = 0
    original_weight_total = 0
    neutral_weight_total = 0
    needs_review_count = 0
    for candidate in candidates:
        summary = candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {}
        if not summary:
            continue
        dominant = str(summary.get("dominant_side") or "unknown")
        by_dominant_side[dominant] = by_dominant_side.get(dominant, 0) + 1
        candidate_weight_total += int(summary.get("candidate_weight") or 0)
        original_weight_total += int(summary.get("original_weight") or 0)
        neutral_weight_total += int(summary.get("neutral_weight") or 0)
        if summary.get("needs_review_by_source_vote"):
            needs_review_count += 1
        for source in summary.get("supports_candidate") or []:
            key = str(source or "unknown")
            by_candidate_support_source[key] = by_candidate_support_source.get(key, 0) + 1
        for source in summary.get("supports_original") or []:
            key = str(source or "unknown")
            by_original_support_source[key] = by_original_support_source.get(key, 0) + 1
        if summary.get("has_source_conflict") or summary.get("needs_review_by_source_vote"):
            conflict_rows.append({
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "correction_type": str(candidate.get("correction_type") or ""),
                "time_range": str(candidate.get("time_range") or ""),
                "original_text": str(candidate.get("original_text") or "")[:120],
                "candidate_text": str(candidate.get("candidate_text") or candidate.get("suggested_text") or "")[:120],
                "dominant_side": dominant,
                "candidate_weight": int(summary.get("candidate_weight") or 0),
                "original_weight": int(summary.get("original_weight") or 0),
                "supports_candidate": [str(item) for item in summary.get("supports_candidate") or [] if str(item)],
                "supports_original": [str(item) for item in summary.get("supports_original") or [] if str(item)],
                "strong_candidate_sources": [str(item) for item in summary.get("strong_candidate_sources") or [] if str(item)],
                "strong_original_sources": [str(item) for item in summary.get("strong_original_sources") or [] if str(item)],
                "needs_review_by_source_vote": bool(summary.get("needs_review_by_source_vote")),
            })
    conflict_rows.sort(key=lambda row: (not row.get("needs_review_by_source_vote"), -abs(int(row.get("candidate_weight") or 0) - int(row.get("original_weight") or 0)), str(row.get("candidate_id") or "")))
    return {
        "candidate_count_with_votes": sum(by_dominant_side.values()),
        "source_conflict_count": len(conflict_rows),
        "needs_review_by_source_vote_count": needs_review_count,
        "candidate_weight_total": candidate_weight_total,
        "original_weight_total": original_weight_total,
        "neutral_weight_total": neutral_weight_total,
        "by_dominant_side": dict(sorted(by_dominant_side.items())),
        "by_candidate_support_source": dict(sorted(by_candidate_support_source.items())),
        "by_original_support_source": dict(sorted(by_original_support_source.items())),
        "conflict_preview": conflict_rows[:12],
    }

def _semantic_correction_ui_summary(
    pack: Any,
    validation: Any,
    closure: Any,
    detail_summary: dict[str, Any],
    candidate_discovery: dict[str, Any],
    review_closure_summary: dict[str, Any],
    *,
    status: str,
    next_action_key: str,
    readable_impact: Any,
    summary_impact: Any,
) -> dict[str, Any]:
    candidates = [row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    accepted = [row for row in (validation.get("accepted_decisions") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    no_change = [row for row in (validation.get("arbitrated_no_change_decisions") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    rejected = [row for row in (validation.get("rejected_decisions") if isinstance(validation, dict) else []) or [] if isinstance(row, dict)]
    applied = [row for row in (closure.get("applied_corrections") if isinstance(closure, dict) else []) or [] if isinstance(row, dict)]
    human_review_candidates = [row for row in candidates if row.get("needs_human_review") or str(row.get("risk_level") or "") == "high"]
    auto_candidate_count = max(0, len(candidates) - len(human_review_candidates))
    accepted_type_counts = _count_values(row.get("correction_type") for row in accepted)
    applied_type_counts = _count_values(row.get("correction_type") for row in applied)
    rejected_reason_counts: dict[str, int] = {}
    for row in rejected:
        for reason in row.get("reject_reasons") or row.get("rejection_reasons") or []:
            key = str(reason or "unknown")
            rejected_reason_counts[key] = rejected_reason_counts.get(key, 0) + 1
    applied_preview = []
    for row in applied[:8]:
        applied_preview.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "correction_type": str(row.get("correction_type") or ""),
                "original_text": str(row.get("original_text") or "")[:120],
                "corrected_text": str(row.get("corrected_text") or "")[:120],
                "confidence": row.get("confidence"),
                "segment_index": row.get("segment_index"),
            }
        )
    candidate_discovery_summary = {
        "status": candidate_discovery.get("status"),
        "next_action_key": candidate_discovery.get("next_action_key"),
        "segment_count": int(candidate_discovery.get("segment_count") or 0),
        "suggestion_count": int(candidate_discovery.get("suggestion_count") or 0),
        "imported_candidate_count": int(candidate_discovery.get("imported_candidate_count") or 0),
        "skipped_count": int(candidate_discovery.get("skipped_count") or 0),
    }
    readable_status = str(readable_impact.get("status") or "missing") if isinstance(readable_impact, dict) else "missing"
    summary_status = str(summary_impact.get("status") or "missing") if isinstance(summary_impact, dict) else "missing"
    if status == "impact_passed":
        ui_state = "closed_and_export_checked"
    elif status == "arbitrated_no_change":
        ui_state = "closed_without_text_changes"
    elif accepted and not applied:
        ui_state = "accepted_waiting_for_closure"
    elif detail_summary.get("review_required_items"):
        ui_state = "human_review_required"
    elif candidates:
        ui_state = "machine_review_required"
    else:
        ui_state = "no_candidates"
    return {
        "ui_state": ui_state,
        "next_action_key": next_action_key,
        "candidate_count": len(candidates),
        "auto_candidate_count": auto_candidate_count,
        "human_review_candidate_count": len(human_review_candidates),
        "candidate_type_counts": detail_summary.get("candidate_type_counts") or {},
        "risk_level_counts": detail_summary.get("risk_level_counts") or {},
        "evidence_source_counts": detail_summary.get("evidence_source_counts") or {},
        "accepted_decision_count": len(accepted),
        "accepted_decision_type_counts": accepted_type_counts,
        "arbitrated_no_change_count": len(no_change),
        "rejected_decision_count": len(rejected),
        "rejected_decision_reason_counts": rejected_reason_counts,
        "applied_correction_count": len(applied),
        "applied_correction_type_counts": applied_type_counts,
        "applied_correction_preview": applied_preview,
        "review_required_count": len(detail_summary.get("review_required_items") or []),
        "review_closed_count": int(review_closure_summary.get("closed_review_decision_count") or 0),
        "review_imported_count": int(review_closure_summary.get("imported_review_decision_count") or 0),
        "candidate_discovery": candidate_discovery_summary,
        "export_chain": {
            "readable_impact_status": readable_status,
            "summary_impact_status": summary_status,
            "corrected_transcript_ready": bool(isinstance(closure, dict) and closure.get("applied_correction_count")),
        },
    }
def _semantic_attention_items(candidates: list[dict[str, Any]], chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus_types = {"number", "action", "concept", "ordinary_word", "ordinary_homophone", "collocation_anomaly", "punctuation", "segment_boundary"}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        kind = str(candidate.get("correction_type") or "")
        if kind not in focus_types and not candidate.get("needs_human_review") and not candidate.get("has_conflict"):
            continue
        start = _float(candidate.get("start", 0.0))
        end = _float(candidate.get("end", start), start)
        chapter = _chapter_for_semantic_time(chapters, start, end)
        priority_score = _semantic_attention_priority(candidate)
        rows.append({
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "candidate_group_id": str(candidate.get("candidate_group_id") or ""),
            "correction_type": kind,
            "risk_level": str(candidate.get("risk_level") or "unknown"),
            "priority_score": priority_score,
            "reason": str(candidate.get("reason") or ""),
            "time_range": str(candidate.get("time_range") or ""),
            "start": start,
            "end": end,
            "chapter_index": chapter.get("chapter_index"),
            "chapter_title": chapter.get("chapter_title", ""),
            "original_text": str(candidate.get("original_text") or "")[:160],
            "suggested_text": str(candidate.get("suggested_text") or candidate.get("candidate_text") or candidate.get("canonical_hint") or "")[:160],
            "evidence_source_types": [str(item) for item in candidate.get("evidence_source_types") or [] if str(item)],
            "source_support_summary": candidate.get("source_support_summary") if isinstance(candidate.get("source_support_summary"), dict) else {},
            "needs_human_review": bool(candidate.get("needs_human_review")),
            "has_conflict": bool(candidate.get("has_conflict")),
        })
    rows.sort(key=lambda row: (-int(row.get("priority_score") or 0), _float(row.get("start", 0.0)), str(row.get("candidate_id") or "")))
    return rows


def _semantic_attention_priority(candidate: dict[str, Any]) -> int:
    risk_rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    type_rank = {"number": 6, "action": 5, "concept": 4, "ordinary_homophone": 4, "collocation_anomaly": 4, "ordinary_word": 3, "segment_boundary": 3, "punctuation": 2}
    score = risk_rank.get(str(candidate.get("risk_level") or "unknown"), 0) * 10
    score += type_rank.get(str(candidate.get("correction_type") or ""), 0)
    if candidate.get("has_conflict"):
        score += 4
    if candidate.get("needs_human_review"):
        score += 3
    source_count = len(candidate.get("evidence_source_types") or [])
    score += min(4, source_count)
    if str(candidate.get("reason") or "") == "deictic_or_low_information_transcript_with_support_concept":
        score += 3
    return score


def _load_semantic_chapter_ranges(root: Path) -> list[dict[str, Any]]:
    payload = _read_optional_json(root / "exports" / "smart-summary-chapters.json")
    chapters = payload.get("chapters") if isinstance(payload, dict) and isinstance(payload.get("chapters"), list) else []
    rows: list[dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        start = _float(chapter.get("start", 0.0))
        end = _float(chapter.get("end", start), start)
        index = int(chapter.get("index") or chapter.get("chapter_index") or len(rows) + 1)
        rows.append(
            {
                "chapter_index": index,
                "chapter_title": str(chapter.get("title") or f"Chapter {index}"),
                "chapter_time_range": f"{chapter.get('start_time') or format_timestamp(start)} - {chapter.get('end_time') or format_timestamp(end)}",
                "start": start,
                "end": end,
            }
        )
    return rows


def _chapter_for_semantic_time(chapters: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    if not chapters:
        return {"chapter_index": 0, "chapter_title": "unassigned", "chapter_time_range": ""}
    midpoint = start + max(0.0, end - start) / 2.0
    for chapter in chapters:
        if _float(chapter.get("start")) <= midpoint <= _float(chapter.get("end")):
            return chapter
    for chapter in chapters:
        if _float(chapter.get("end")) >= start and _float(chapter.get("start")) <= end:
            return chapter
    return chapters[-1]


def _semantic_chapter_risk_summary(candidates: list[dict[str, Any]], review_items: list[dict[str, Any]], chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}

    def ensure_group(chapter: dict[str, Any]) -> dict[str, Any]:
        index = int(chapter.get("chapter_index") or 0)
        if index not in grouped:
            grouped[index] = {
                "chapter_index": index,
                "chapter_title": chapter.get("chapter_title") or "unassigned",
                "chapter_time_range": chapter.get("chapter_time_range") or "",
                "candidate_count": 0,
                "review_required_count": 0,
                "risk_level_counts": {},
                "candidate_type_counts": {},
                "evidence_source_counts": {},
                "high_risk_candidate_ids": [],
                "review_required_candidate_ids": [],
            }
        return grouped[index]

    for candidate in candidates:
        start = _float(candidate.get("start", 0.0))
        end = _float(candidate.get("end", start), start)
        chapter = _chapter_for_semantic_time(chapters, start, end)
        group = ensure_group(chapter)
        group["candidate_count"] += 1
        risk = str(candidate.get("risk_level") or "unknown")
        kind = str(candidate.get("correction_type") or "unknown")
        group["risk_level_counts"][risk] = group["risk_level_counts"].get(risk, 0) + 1
        group["candidate_type_counts"][kind] = group["candidate_type_counts"].get(kind, 0) + 1
        for source_type in candidate.get("evidence_source_types") or []:
            key = str(source_type or "unknown")
            group["evidence_source_counts"][key] = group["evidence_source_counts"].get(key, 0) + 1
        if risk == "high" or candidate.get("needs_human_review"):
            group["high_risk_candidate_ids"].append(str(candidate.get("candidate_id") or ""))

    for item in review_items:
        chapter = {"chapter_index": item.get("chapter_index") or 0, "chapter_title": item.get("chapter_title") or "unassigned", "chapter_time_range": item.get("chapter_time_range") or ""}
        group = ensure_group(chapter)
        group["review_required_count"] += 1
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            group["review_required_candidate_ids"].append(candidate_id)

    return [grouped[key] for key in sorted(grouped)]


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts
def _semantic_review_closure_summary(root: Path, validation: Any, detail_summary: dict[str, Any]) -> dict[str, Any]:
    imported = _read_optional_json(root / "transcript-semantic-correction-result.review.json")
    decisions = [row for row in (imported.get("decisions") if isinstance(imported, dict) else []) or [] if isinstance(row, dict)]
    accepted_ids = {
        str(row.get("candidate_id"))
        for row in (validation.get("accepted_decisions") if isinstance(validation, dict) else []) or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    open_items = [row for row in detail_summary.get("review_required_items") or [] if isinstance(row, dict)]
    open_ids = {str(row.get("candidate_id")) for row in open_items if str(row.get("candidate_id") or "")}
    action_counts = _count_values(row.get("action") for row in decisions)
    import_summary = imported.get("import_summary") if isinstance(imported.get("import_summary"), dict) else {}
    imported_ids = {str(row.get("candidate_id") or "") for row in decisions if str(row.get("candidate_id") or "")}
    accepted_imported_ids = accepted_ids & imported_ids
    rejected_imported_ids = {
        str(row.get("candidate_id") or "")
        for row in (validation.get("rejected_decisions") if isinstance(validation, dict) else []) or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "") in imported_ids
    }
    closed_ids: list[str] = []
    still_open_after_import: list[str] = []
    for row in decisions:
        candidate_id = str(row.get("candidate_id") or "")
        action = str(row.get("action") or "")
        if not candidate_id:
            continue
        if action in {"replace", "keep_original"} and (candidate_id in accepted_ids or action == "keep_original") and candidate_id not in open_ids:
            closed_ids.append(candidate_id)
        else:
            still_open_after_import.append(candidate_id)
    open_candidate_ids = sorted(open_ids | set(still_open_after_import))
    if accepted_imported_ids:
        next_action_key = "run_transcript_semantic_correction_closure"
        next_action_command = f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure '{root}' --input-json '{root / 'transcript-semantic-correction-result.review.json'}' --refresh-exports"
    elif decisions:
        next_action_key = "fix_review_notes_or_collect_more_evidence"
        next_action_command = f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-review-notes '{root}' --review-json '{root / 'transcript-semantic-correction-review-notes.json'}'"
    else:
        next_action_key = "prepare_semantic_review_notes"
        next_action_command = f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-review-notes '{root}' --review-json '{root / 'transcript-semantic-correction-review-notes.json'}'"
    return {
        "review_result_imported": bool(decisions),
        "imported_review_decision_count": len(decisions),
        "accepted_imported_review_decision_count": len(accepted_imported_ids),
        "rejected_imported_review_decision_count": len(rejected_imported_ids),
        "skipped_review_note_count": int(import_summary.get("skipped_count") or 0),
        "closed_review_decision_count": len(sorted(set(closed_ids))),
        "open_review_required_count": len(open_items),
        "validation_status": str(validation.get("status") or "missing") if isinstance(validation, dict) else "missing",
        "next_action_key": next_action_key,
        "next_action_command": next_action_command,
        "actions": action_counts,
        "closed_candidate_ids": sorted(set(closed_ids)),
        "open_candidate_ids": open_candidate_ids,
        "result_json": str(root / "transcript-semantic-correction-result.review.json"),
        "result_markdown": str(root / "transcript-semantic-correction-result.review.md"),
        "skipped": import_summary.get("skipped") if isinstance(import_summary.get("skipped"), list) else [],
    }
def _semantic_review_count(root: Path, validation: Any) -> int:
    if isinstance(validation, dict):
        return int(validation.get("review_required_count") or len(validation.get("review_rows") or []))
    review = _read_optional_json(root / "transcript-semantic-correction-review.json")
    if isinstance(review, dict):
        return int(review.get("review_required_count") or len(review.get("items") or []))
    return 0


def _short_evidence_excerpt(candidate: dict[str, Any]) -> str:
    parts = []
    for item in candidate.get("evidence", [])[:4]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_type") or "evidence")
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(f"{source}: {text[:160]}")
    return " | ".join(parts)


def _accepted_correction_decisions(root: Path) -> list[dict[str, Any]]:
    ledger = _read_optional_json(root / "transcript-semantic-correction-decision-ledger.json")
    ledger_rows = ledger.get("decisions") if isinstance(ledger, dict) else []
    if isinstance(ledger_rows, list) and ledger_rows:
        return [dict(row) for row in ledger_rows if isinstance(row, dict)]
    validation = _read_optional_json(root / "transcript-semantic-correction-validation.json")
    rows = validation.get("accepted_decisions") if isinstance(validation, dict) else []
    return [dict(row) for row in (rows or []) if isinstance(row, dict)]


def _ledger_decision_key(decision: dict[str, Any]) -> tuple[Any, ...]:
    return (
        decision.get("segment_index"),
        str(decision.get("correction_type") or ""),
        str(decision.get("original_text") or ""),
        str(decision.get("apply_scope") or "segment"),
        tuple(int(value) for value in (decision.get("merge_segment_indexes") or []) if isinstance(value, int)),
    )


def _ledger_decision_row(decision: dict[str, Any], *, validation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "action",
        "segment_index",
        "correction_type",
        "original_text",
        "corrected_text",
        "confidence",
        "semantic_rationale",
        "rationale",
        "evidence_ids",
        "human_confirmed",
        "safe_to_apply",
        "needs_human_review",
        "apply_scope",
        "segments",
        "merge_segment_indexes",
    )
    row = {key: decision.get(key) for key in keys if key in decision}
    row["accepted_at"] = now_iso()
    row["validation_pack_sha256"] = str(validation.get("pack_sha256") or "")
    row["validation_input_json"] = str(validation.get("input_json") or "")
    return row


def _merge_correction_decision_ledger(
    root: Path,
    accepted: list[dict[str, Any]],
    *,
    validation: dict[str, Any],
) -> dict[str, Any]:
    existing = _read_optional_json(root / "transcript-semantic-correction-decision-ledger.json")
    rows = [dict(row) for row in (existing.get("decisions") if isinstance(existing, dict) else []) or [] if isinstance(row, dict)]
    by_key = {_ledger_decision_key(row): index for index, row in enumerate(rows)}
    for decision in accepted:
        incoming = _ledger_decision_row(decision, validation=validation)
        key = _ledger_decision_key(incoming)
        prior_index = by_key.get(key)
        if prior_index is None:
            by_key[key] = len(rows)
            rows.append(incoming)
            continue
        prior = rows[prior_index]
        if str(prior.get("corrected_text") or "") == str(incoming.get("corrected_text") or ""):
            continue
        if not bool(incoming.get("human_confirmed")):
            raise ValueError("correction_decision_conflict_requires_human_confirmation")
        rows[prior_index] = incoming
    return {
        "schema": DECISION_LEDGER_SCHEMA,
        "bundle_dir": str(root),
        "decision_count": len(rows),
        "decisions": rows,
        "updated_at": now_iso(),
    }


def apply_human_confirmed_source_fidelity_decisions(
    cues: list[Any],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply exact, segment-scoped human corrections through the canonical engine.

    Intent: let evaluation-only reference windows use corrections the user
    confirmed against this recording without creating a second replacement
    implementation.
    Decision: validate the narrow human-confirmed term contract, then delegate
    to the existing semantic correction applier.
    Reason: broad/global replacements or boundary changes would make an A/B
    reference less auditable and could silently alter other recordings.
    Evidence: the current consultation has exact corrections for four ASR
    phrases, each bound to a source segment and anonymous speaker.
    Effective scope: caller-provided cues only; no Bundle, ledger, canonical
    transcript, speaker role, global dictionary, or provider input is changed.
    """

    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            raise ValueError(f"human correction {index} must be an object")
        if decision.get("human_confirmed") is not True:
            raise ValueError(f"human correction {index} is not human_confirmed")
        if decision.get("action") != "replace":
            raise ValueError(f"human correction {index} must use action=replace")
        if not isinstance(decision.get("segment_index"), int):
            raise ValueError(f"human correction {index} requires an exact segment_index")
        if str(decision.get("apply_scope") or "segment").casefold() != "segment":
            raise ValueError(f"human correction {index} must be segment-scoped")
        if str(decision.get("correction_type") or "term") in {
            "punctuation",
            "segment_boundary",
        }:
            raise ValueError(f"human correction {index} cannot change segment structure")
        original = str(decision.get("original_text") or "")
        corrected = str(decision.get("corrected_text") or "")
        if not original or not corrected or original == corrected:
            raise ValueError(f"human correction {index} requires distinct text values")
    return _apply_decisions_to_cues(cues, decisions)


def _apply_decisions_to_cues(cues: list[Any], decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    speaker_labels = speaker_label_map(cues)
    global_replace_decisions = [
        row
        for row in decisions
        if row.get("action") == "replace"
        and str(row.get("apply_scope") or "").lower() in {"all_segments", "global"}
        and str(row.get("correction_type") or "") not in {"punctuation", "segment_boundary"}
    ]
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        seg = decision.get("segment_index")
        if isinstance(seg, int):
            by_segment.setdefault(seg, []).append(decision)
    applied: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    skipped_merge_indexes: set[int] = set()
    for idx, cue in enumerate(cues):
        if idx in skipped_merge_indexes:
            continue
        raw_text = str(getattr(cue, "text", "") or "")
        merge_decision = _first_merge_decision(by_segment.get(idx, []))
        if merge_decision:
            merge_indexes = [index for index in merge_decision.get("merge_segment_indexes", []) if isinstance(index, int) and 0 <= index < len(cues)]
            merge_cues = [cues[index] for index in merge_indexes]
            merge_speakers = {cue_speaker(item) for item in merge_cues if cue_speaker(item)}
            if len(merge_speakers) > 1:
                raise ValueError("semantic_correction_merge_crosses_speaker_boundary")
            raw_parts = [str(getattr(item, "text", "") or "") for item in merge_cues]
            raw_merged_text = " ".join(part for part in raw_parts if part).strip()
            corrected_text = str(merge_decision.get("corrected_text") or raw_merged_text).strip()
            item = {
                "candidate_id": merge_decision.get("candidate_id"),
                "correction_type": merge_decision.get("correction_type"),
                "original_text": raw_merged_text,
                "corrected_text": corrected_text,
                "confidence": merge_decision.get("confidence"),
                "rationale": merge_decision.get("rationale"),
                "application": "segment_merge",
                "source_segment_indexes": merge_indexes,
                "merged_segment_count": len(merge_indexes),
            }
            segments.append(
                {
                    "index": len(segments),
                    "source_segment_indexes": merge_indexes,
                    "start": _float(getattr(merge_cues[0], "start", 0.0)),
                    "end": _float(getattr(merge_cues[-1], "end", 0.0)),
                    **speaker_payload(merge_cues[0], speaker_labels),
                    "metadata": dict(getattr(merge_cues[0], "metadata", {}) or {}),
                    "text": corrected_text,
                    "raw_text": raw_merged_text,
                    "corrected_text": corrected_text,
                    "changed": True,
                    "structure_changed": True,
                    "semantic_corrections": [item],
                }
            )
            applied.append({**item, "segment_index": idx})
            skipped_merge_indexes.update(index for index in merge_indexes if index != idx)
            continue
        split_decision = _first_split_decision(by_segment.get(idx, []))
        if split_decision:
            split_rows = split_decision.get("segments") if isinstance(split_decision.get("segments"), list) else []
            item = {
                "candidate_id": split_decision.get("candidate_id"),
                "correction_type": split_decision.get("correction_type"),
                "original_text": raw_text,
                "corrected_text": " ".join(str(row.get("text") or "").strip() for row in split_rows),
                "confidence": split_decision.get("confidence"),
                "rationale": split_decision.get("rationale"),
                "application": "segment_split",
                "split_segment_count": len(split_rows),
            }
            for split_index, row in enumerate(split_rows):
                correction = {**item, "split_segment_index": split_index}
                segments.append(
                    {
                        "index": len(segments),
                        "source_segment_index": idx,
                        "split_segment_index": split_index,
                        "start": _float(row.get("start", getattr(cue, "start", 0.0))),
                        "end": _float(row.get("end", getattr(cue, "end", 0.0))),
                        **speaker_payload(cue, speaker_labels),
                        "metadata": dict(getattr(cue, "metadata", {}) or {}),
                        "text": str(row.get("text") or ""),
                        "raw_text": raw_text,
                        "corrected_text": str(row.get("text") or ""),
                        "changed": True,
                        "structure_changed": True,
                        "semantic_corrections": [correction],
                    }
                )
            applied.append({**item, "segment_index": idx})
            continue
        text = raw_text
        corrections = []
        segment_decisions = [*global_replace_decisions, *by_segment.get(idx, [])]
        seen_decision_keys: set[tuple[str, str, str]] = set()
        for decision in segment_decisions:
            dedupe_key = (str(decision.get("candidate_id") or ""), str(decision.get("original_text") or ""), str(decision.get("corrected_text") or ""))
            if dedupe_key in seen_decision_keys:
                continue
            seen_decision_keys.add(dedupe_key)
            if decision.get("action") != "replace":
                continue
            original = str(decision.get("original_text") or "")
            corrected = str(decision.get("corrected_text") or "")
            kind = str(decision.get("correction_type") or "")
            if kind in {"punctuation", "segment_boundary"}:
                if corrected and corrected != text:
                    previous_text = text
                    text = corrected
                    item = {"candidate_id": decision.get("candidate_id"), "correction_type": kind, "original_text": previous_text, "corrected_text": corrected, "confidence": decision.get("confidence"), "rationale": decision.get("rationale"), "application": "whole_segment_text"}
                    corrections.append(item)
                    applied.append({**item, "segment_index": idx})
                continue
            if original and corrected and original in text:
                text = text.replace(original, corrected)
                item = {"candidate_id": decision.get("candidate_id"), "correction_type": kind or decision.get("correction_type"), "original_text": original, "corrected_text": corrected, "confidence": decision.get("confidence"), "rationale": decision.get("rationale"), "application": "substring_replace", "apply_scope": decision.get("apply_scope") or "segment"}
                corrections.append(item)
                applied.append({**item, "segment_index": idx})
            elif original and corrected:
                normalized_replacement = _replace_normalized_occurrences(text, original, corrected)
                application = "normalized_substring_replace"
                if normalized_replacement == text and _normalize_compact(corrected) in _normalize_compact(text):
                    normalized_replacement = text
                    application = "already_present"
                if normalized_replacement != text or application == "already_present":
                    text = normalized_replacement
                    item = {"candidate_id": decision.get("candidate_id"), "correction_type": kind or decision.get("correction_type"), "original_text": original, "corrected_text": corrected, "confidence": decision.get("confidence"), "rationale": decision.get("rationale"), "application": application, "apply_scope": decision.get("apply_scope") or "segment"}
                    corrections.append(item)
                    applied.append({**item, "segment_index": idx})
        segments.append(
            {
                "index": len(segments),
                "source_segment_index": idx,
                "start": _float(getattr(cue, "start", 0.0)),
                "end": _float(getattr(cue, "end", 0.0)),
                **speaker_payload(cue, speaker_labels),
                "metadata": dict(getattr(cue, "metadata", {}) or {}),
                "text": text,
                "raw_text": raw_text,
                "corrected_text": text,
                "changed": text != raw_text,
                "structure_changed": any(row.get("application") == "whole_segment_text" for row in corrections),
                "semantic_corrections": corrections,
            }
        )
    return segments, applied

def _replace_normalized_occurrences(text: str, original: str, corrected: str) -> str:
    """Replace a segment-bound phrase after harmless spacing/punctuation cleanup."""
    tokens = re.findall(r"[0-9A-Za-z\u3400-\u9fff]+", original)
    if not tokens or _normalize_compact(original) not in _normalize_compact(text):
        return text
    separator = r"[^0-9A-Za-z\u3400-\u9fff]*"
    pattern = separator.join(re.escape(token) for token in tokens)
    return re.sub(pattern, lambda _match: corrected, text, flags=re.IGNORECASE)


def _first_merge_decision(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for decision in decisions:
        if decision.get("action") != "replace":
            continue
        if str(decision.get("correction_type") or "") not in {"segment_boundary", "punctuation"}:
            continue
        indexes = decision.get("merge_segment_indexes") if isinstance(decision.get("merge_segment_indexes"), list) else []
        if len(indexes) >= 2 and indexes[0] == decision.get("segment_index"):
            return decision
    return None


def _first_split_decision(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for decision in decisions:
        if decision.get("action") != "replace":
            continue
        if str(decision.get("correction_type") or "") not in {"segment_boundary", "punctuation"}:
            continue
        segments = decision.get("segments") if isinstance(decision.get("segments"), list) else []
        if len(segments) >= 2:
            return decision
    return None


def _write_closure(
    root: Path,
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    corrected_payload: dict[str, Any] | None,
    decision_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Validation updates the manifest before closure starts. Reload it here so
    # the transaction never overwrites those concurrent-in-sequence pointers
    # with the stale object read at the beginning of the public operation.
    del manifest
    manifest = _read_manifest(root)
    manifest_path = root / "manifest.json"
    pack_path = root / "transcript-semantic-correction-pack.json"
    validation_path = root / "transcript-semantic-correction-validation.json"
    input_path = Path(str(result.get("input_json") or "")).expanduser().resolve()
    required_inputs = {
        "manifest.json": manifest_path,
        "transcript-semantic-correction-pack.json": pack_path,
        "transcript-semantic-correction-validation.json": validation_path,
    }
    for label, path in required_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"semantic_closure_required_input_missing:{label}")
    if not input_path.is_file():
        raise FileNotFoundError(f"semantic_closure_result_input_missing:{input_path}")
    try:
        input_relative = input_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"semantic_closure_result_input_outside_bundle:{input_path}") from exc

    pack_sha256 = _sha256_file(pack_path)
    validation_sha256 = _sha256_file(validation_path)
    input_sha256 = _sha256_file(input_path)
    input_identity = {
        "path": str(input_path),
        "relative_path": input_relative,
        "sha256": input_sha256,
    }
    result["pack_sha256"] = pack_sha256
    result["validation_sha256"] = validation_sha256
    result["input_identity"] = input_identity

    canonical_path = root / "source-arbitrated-transcript.json"
    canonical_before_sha256 = _sha256_file(canonical_path)
    canonical_after_sha256 = canonical_before_sha256
    artifacts: dict[str, str] = {}
    if decision_ledger is not None:
        artifacts["transcript-semantic-correction-decision-ledger.json"] = _json_artifact_payload(decision_ledger)
        manifest["transcript_semantic_correction_decision_ledger_json"] = "transcript-semantic-correction-decision-ledger.json"
    if corrected_payload is not None:
        semantic = corrected_payload.setdefault("semantic_correction", {})
        if isinstance(semantic, dict):
            semantic["pack_sha256"] = pack_sha256
            semantic["validation_sha256"] = validation_sha256
            semantic["input_identity"] = input_identity
        canonical_json = _json_artifact_payload(corrected_payload)
        canonical_unchanged = _canonical_payload_equivalent(
            _read_optional_json(canonical_path),
            corrected_payload,
        )
        derived_exist = all(
            (root / name).is_file()
            for name in (
                "source-arbitrated-transcript.srt",
                "source-arbitrated-transcript.md",
            )
        )
        needs_canonical_write = not canonical_unchanged or not derived_exist
        result["canonical_write_status"] = "written" if needs_canonical_write else "unchanged"
        if needs_canonical_write:
            artifacts["source-arbitrated-transcript.json"] = canonical_json
            artifacts["source-arbitrated-transcript.srt"] = _render_srt(corrected_payload.get("segments") or [])
            artifacts["source-arbitrated-transcript.md"] = _render_corrected_markdown(corrected_payload)
            canonical_after_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
        manifest["source_arbitrated_transcript_srt"] = "source-arbitrated-transcript.srt"
        manifest["source_arbitrated_transcript_markdown"] = "source-arbitrated-transcript.md"
        manifest["corrected_transcript_json"] = "source-arbitrated-transcript.json"
        manifest["corrected_transcript_srt"] = "source-arbitrated-transcript.srt"
        manifest["corrected_transcript_markdown"] = "source-arbitrated-transcript.md"
        manifest["corrected_transcript_source"] = "transcript_semantic_correction"
    result["canonical_before_sha256"] = canonical_before_sha256
    result["canonical_after_sha256"] = canonical_after_sha256
    result["transaction_contract"] = {
        "atomic_closure_artifacts": True,
        "expected_pack_sha256": pack_sha256,
        "expected_validation_sha256": validation_sha256,
        "expected_input_identity": input_identity,
        "raw_asr_untouched": True,
        "timeline_untouched": True,
    }

    closure_args = {
        "bundle_dir": str(root),
        "input_json": result.get("input_json", ""),
        "min_confidence": result.get("min_confidence", 0.88),
        "auto_apply": result.get("auto_apply", False),
        "refresh_exports": result.get("refresh_exports_requested", False),
        "write": True,
    }
    manifest["transcript_semantic_correction_closure_json"] = "transcript-semantic-correction-closure.json"
    manifest["transcript_semantic_correction_closure_markdown"] = "transcript-semantic-correction-closure.md"
    manifest["mcp_transcript_semantic_correction_closure_args"] = "mcp-transcript-semantic-correction-closure.args.json"
    manifest["transcript_semantic_correction_closure_summary"] = {
        "status": result.get("status"),
        "applied_correction_count": result.get("applied_correction_count", 0),
        "changed_segment_count": result.get("changed_segment_count", 0),
        "refresh_exports_status": result.get("refresh_exports_status"),
        "pack_sha256": pack_sha256,
        "validation_sha256": validation_sha256,
        "input_identity": input_identity,
        "transaction_status": "committed",
        "updated_at": result.get("updated_at"),
    }
    artifacts.update(
        {
            "transcript-semantic-correction-closure.json": _json_artifact_payload(result),
            "transcript-semantic-correction-closure.md": _render_closure_markdown(result),
            "mcp-transcript-semantic-correction-closure.args.json": _json_artifact_payload(closure_args),
            "manifest.json": _json_artifact_payload(manifest),
        }
    )
    expected_inputs = {
        "manifest.json": _sha256_file(manifest_path),
        "transcript-semantic-correction-pack.json": pack_sha256,
        "transcript-semantic-correction-validation.json": validation_sha256,
        input_relative: input_sha256,
    }
    return commit_transcript_semantic_artifacts(
        root,
        artifacts=artifacts,
        expected_inputs=expected_inputs,
        operation_identity={
            "operation": "transcript_semantic_correction_closure",
            "closure_status": str(result.get("status") or ""),
            "pack_sha256": pack_sha256,
            "validation_sha256": validation_sha256,
            "input_sha256": input_sha256,
            "input_relative_path": input_relative,
        },
    )


def _json_artifact_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _canonical_payload_equivalent(existing: Any, proposed: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    existing_normalized = dict(existing)
    proposed_normalized = dict(proposed)
    existing_normalized.pop("updated_at", None)
    proposed_normalized.pop("updated_at", None)
    return existing_normalized == proposed_normalized

def _closure_result(root: Path, manifest: dict[str, Any], validation: dict[str, Any], segments: list[dict[str, Any]], applied: list[dict[str, Any]], *, status: str, auto_apply: bool) -> dict[str, Any]:
    changed = sum(1 for row in segments if row.get("changed"))
    return {"schema": CLOSURE_SCHEMA, "bundle_dir": str(root), "title": str(manifest.get("title") or root.name), "status": status, "ok": status in {"completed", "completed_no_text_changes"}, "input_json": str(validation.get("input_json") or ""), "min_confidence": validation.get("min_confidence", 0.88), "auto_apply": bool(auto_apply), "accepted_decision_count": validation.get("accepted_decision_count", 0), "arbitrated_no_change_count": validation.get("arbitrated_no_change_count", 0), "applied_correction_count": len(applied), "changed_segment_count": changed, "applied_corrections": applied, "artifacts": {"json": str(root / "transcript-semantic-correction-closure.json"), "markdown": str(root / "transcript-semantic-correction-closure.md"), "corrected_json": str(root / "source-arbitrated-transcript.json"), "corrected_markdown": str(root / "source-arbitrated-transcript.md")}, "operator_boundary": {"local_only": True, "no_cloud_call": True, "does_not_modify_raw_sources": True}, "updated_at": now_iso()}


def _load_import(input_json: str | Path) -> dict[str, Any]:
    path = Path(input_json).expanduser()
    text = path.read_text(encoding="utf-8-sig") if path.exists() else str(input_json)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = extract_json_document(text)
    if not isinstance(data, dict):
        raise ValueError("semantic correction input must be a JSON object or Markdown containing one")
    return data


def _load_output_documents(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for key, fallback in FINAL_OUTPUT_CANDIDATES:
        path = _bundle_path(root, manifest.get(key) or fallback)
        if not path.exists():
            continue
        if key == "source_arbitrated_transcript_json":
            data = _read_optional_json(path)
            if isinstance(data, dict):
                text = "\n".join(str(row.get("text") or row.get("corrected_text") or "") for row in data.get("segments", []) if isinstance(row, dict))
            else:
                text = ""
        else:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"key": key, "path": str(path), "text": text})
    return docs



def _load_readable_output_documents(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("full_transcript", manifest.get("knowledge_note_transcript_markdown") or manifest.get("full_transcript") or "exports/full-transcript.md", "required_readable_transcript"),
        ("full_body", manifest.get("knowledge_note_full_body_markdown") or manifest.get("full_body") or "exports/full-body.md", "required_readable_transcript"),
        ("smart_summary", manifest.get("smart_summary") or "exports/smart-summary.md", "required_readable_summary"),
        ("knowledge_note", manifest.get("knowledge_note_markdown") or "exports/knowledge-note.md", "reported_audit_note"),
    ]
    docs: list[dict[str, Any]] = []
    for key, value, role in specs:
        path = _bundle_path(root, value)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"key": key, "path": str(path), "role": role, "text": text})
    if not any(row["key"] == "full_transcript" for row in docs):
        creator_transcript = _creator_readable_transcript(root, manifest)
        if creator_transcript:
            docs.append(creator_transcript)
    return docs


def _creator_readable_transcript(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Verify v0.2's actual HTML/SRT instead of requiring retired legacy files.

    The report remains a candidate. This proves correction propagation only;
    it cannot approve the summary or replace any production quality gate.
    """
    from .creator_report_renderers import bundle_file
    from .smart_summary_input_pack import select_canonical_transcript_path
    from .bilinote_transcript_tools import parse_subtitle_text
    from .file_hash import sha256_file
    try:
        receipt = read_json(root / "exports/creator-report/latest.json")
        if (receipt.get("schema") != "video_knowledge_pipeline.creator_report.v2.receipt"
                or receipt.get("profile") != "analysis-v0.2"
                or receipt.get("status") != "exported_candidate"):
            return None
        artifacts = {row["format"]: row for row in receipt["artifacts"]}
        paths = {}
        for kind in ("projection", "html", "srt"):
            record = artifacts[kind]
            paths[kind] = bundle_file(root, record["path"])
            if sha256_file(paths[kind]) != record["sha256"]:
                return None
        projection = read_json(paths["projection"])
        if projection.get("schema") != "video_knowledge_pipeline.creator_report.v2":
            return None
        canonical = select_canonical_transcript_path(root, manifest)
        source = next(row for row in projection["sources"] if row.get("role") == "canonical_transcript")
        if (canonical is None or Path(source["path"]).resolve() != canonical.resolve()
                or source["sha256"] != sha256_file(canonical)):
            return None
        def rows(cues):
            return [(round(float(cue.start),3), round(float(cue.end),3), str(cue.text).strip())
                    for cue in cues if str(cue.text).strip()]
        expected = rows(parse_transcript(canonical))
        projected = [(round(float(row["start"]),3),round(float(row["end"]),3),str(row["text"]).strip())
                     for row in projection["transcript"] if str(row["text"]).strip()]
        subtitled = rows(parse_subtitle_text(paths["srt"].read_text(encoding="utf-8-sig")))
        if not expected or expected != projected or expected != subtitled:
            return None
        return {"key":"full_transcript", "path":str(paths["html"]),
                "role":"hash_bound_v02_html_and_srt", "candidate_only":True,
                "text":"\n".join(row[2] for row in expected)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, StopIteration):
        return None


def _sample_lines(text: str, needle: str, *, limit: int = 3) -> list[str]:
    if not text or not needle:
        return []
    rows: list[str] = []
    for line in text.splitlines():
        if needle in line:
            rows.append(line.strip()[:300])
            if len(rows) >= limit:
                break
    return rows


def _render_readable_impact_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 转写语义纠错可读文件影响报告",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Accepted decisions: `{result.get('accepted_decision_count')}`",
        f"- Required readable residual total: `{result.get('required_readable_residual_total')}`",
        f"- Missing or empty required documents: `{', '.join(result.get('missing_required_documents') or []) or 'none'}`",
        "",
        "## Documents",
        "",
    ]
    for doc in result.get("documents") or []:
        lines.append(f"- `{doc.get('key')}` ({doc.get('role')}): `{doc.get('path')}`")
    lines.extend(["", "## Corrections", ""])
    for row in result.get("corrections") or []:
        lines.append(f"### `{row.get('candidate_id')}` `{row.get('original_text')}` -> `{row.get('corrected_text')}`")
        lines.append(f"- Required residual: `{row.get('required_readable_residual_count')}`")
        by_doc = row.get("by_document") if isinstance(row.get("by_document"), dict) else {}
        for key, doc in by_doc.items():
            lines.append(f"- `{key}` original={doc.get('original_count')} corrected={doc.get('corrected_count')}")
            samples = doc.get("sample_corrected_lines") or []
            if samples:
                lines.append(f"  - sample: {samples[0]}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in result.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"

def _semantic_candidate_discovery_segments(
    cues: list[Any],
    timeline: list[Any],
    candidates: list[dict[str, Any]],
    *,
    sidecar_sources: list[dict[str, Any]],
    metadata_evidence: list[dict[str, Any]],
    limit: int,
    cursor: int = 0,
) -> list[dict[str, Any]]:
    candidates_by_segment: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        seg = candidate.get("segment_index")
        if str(seg).lstrip("-").isdigit():
            candidates_by_segment.setdefault(int(seg), []).append(candidate)
    rows: list[dict[str, Any]] = []
    for idx, cue in enumerate(cues):
        if idx < max(0, int(cursor or 0)):
            continue
        text = str(getattr(cue, "text", "") or "").strip()
        if not text:
            continue
        start = _float(getattr(cue, "start", 0.0))
        end = _float(getattr(cue, "end", 0.0), start)
        tl_item, timeline_alignment = _timeline_match_for_cue(
            timeline,
            cue,
            start,
            end,
        )
        evidence = _dedupe_evidence([
            *_evidence_for_cue(
                cue,
                tl_item,
                segment_index=idx,
                timeline_alignment=timeline_alignment,
            ),
            *_sidecar_evidence_for_cue(cue, sidecar_sources),
            *metadata_evidence,
        ])
        score, reasons = _semantic_candidate_discovery_score(text, evidence, candidates_by_segment.get(idx, []), start=start, end=end)
        reasons = ["bounded_full_transcript_scan", *reasons]
        rows.append({
            "source_segment_index": idx,
            "start": start,
            "end": end,
            "time_range": f"{format_timestamp(start)} - {format_timestamp(end)}",
            "text": text[:1200],
            "score": score,
            "reasons": reasons,
            "existing_candidate_ids": [str(row.get("candidate_id") or "") for row in candidates_by_segment.get(idx, []) if str(row.get("candidate_id") or "")],
            "evidence": [_compact_discovery_evidence(item) for item in evidence[:8]],
        })
    rows.sort(key=lambda row: int(row.get("source_segment_index") or 0))
    if limit and limit > 0:
        rows = rows[: min(int(limit), DISCOVERY_MAX_SEGMENTS)]
    else:
        rows = rows[:DISCOVERY_MAX_SEGMENTS]
    return rows


def _semantic_candidate_discovery_score(text: str, evidence: list[dict[str, Any]], existing: list[dict[str, Any]], *, start: float = 0.0, end: float = 0.0) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    if existing:
        score += 12
        reasons.append("has_existing_rule_candidates")
    if DEICTIC_OR_SCREEN_REF_RE.search(text):
        score += 16
        reasons.append("deictic_or_screen_reference")
    if _looks_fragmented(text) or FILLER_RE.search(text):
        score += 14
        reasons.append("fragmented_or_semantically_weak_phrase")
    if MOJIBAKE_RE.search(text):
        score += 24
        reasons.append("mojibake_or_encoding_noise")
    boundary_kind = _punctuation_or_boundary_kind(text, start=start, end=end)
    if boundary_kind == "segment_boundary":
        score += 18
        reasons.append("needs_segment_boundary_review")
    elif boundary_kind == "punctuation":
        score += 14
        reasons.append("needs_punctuation_review")
    elif not PUNCTUATION_RE.search(text) and len(text) >= 32:
        score += 8
        reasons.append("long_unpunctuated_asr")
    if _fact_value_markers(text):
        score += 10
        reasons.append("contains_fact_value")
    support_text = " ".join(str(item.get("text") or "") for item in evidence if item.get("source_type") != "asr_or_subtitle")
    if _ordinary_support_diff_candidate(text, support_text) or _visual_conflict_text(text, support_text):
        score += 18
        reasons.append("support_evidence_semantically_differs")
    if any(str(item.get("source_type") or "") in {"ocr", "structured_visual", "visual_understanding", "temporal_visual", "tagger", "platform_subtitle"} for item in evidence):
        score += 6
        reasons.append("has_cross_modal_or_sidecar_evidence")
    return score, reasons


def _compact_discovery_evidence(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "timeline_index": item.get("timeline_index"),
        "start": item.get("start"),
        "end": item.get("end"),
        "text": str(item.get("text") or "")[:700],
        "path": item.get("path") or item.get("frame_path"),
    }


def _candidate_discovery_template(segments: list[dict[str, Any]]) -> dict[str, Any]:
    first = segments[0] if segments else {}
    return {
        "schema": CANDIDATE_SUGGESTIONS_SCHEMA,
        "source": "codex_or_llm_candidate_discovery",
        "suggestions": [
            {
                "source_segment_index": first.get("source_segment_index", 0),
                "start": first.get("start", 0.0),
                "end": first.get("end", 0.0),
                "correction_type": "ordinary_word | ordinary_homophone | collocation_anomaly | proper_noun | term | number | action | concept | negation | condition | exception | person_name | organization | date_time | money | percentage | punctuation | segment_boundary",
                "original_text": "疑似错词原文片段，必须来自该 segment 文本",
                "candidate_text": "可能正确写法；不确定则留空",
                "reason": "为什么它疑似错，不要写成最终结论",
                "confidence": 0.0,
                "evidence_summary": "引用 ASR/字幕/OCR/视觉/打标器/上下文证据",
                "needs_human_review": True,
            }
        ],
    }




def _codex_candidate_discovery_suggestions(discovery: dict[str, Any], *, max_suggestions: int = 40) -> list[dict[str, Any]]:
    segments = [row for row in (discovery.get("segments") or []) if isinstance(row, dict)]
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for segment in segments:
        for suggestion in _codex_suggestions_for_segment(segment):
            key = (
                str(suggestion.get("source_segment_index") or ""),
                str(suggestion.get("original_text") or "").strip().lower(),
                str(suggestion.get("candidate_text") or "").strip().lower(),
            )
            if not key[1] or not key[2] or key in seen:
                continue
            seen.add(key)
            suggestions.append(suggestion)
            if max_suggestions and len(suggestions) >= max_suggestions:
                return suggestions
    return suggestions


def _codex_suggestions_for_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(segment.get("text") or "").strip()
    if not text:
        return []
    reasons = {str(item) for item in segment.get("reasons") or [] if str(item)}
    evidence = [row for row in segment.get("evidence") or [] if isinstance(row, dict)]
    suggestions: list[dict[str, Any]] = []
    spaced = ODD_SPACING_RE.search(text)
    if spaced:
        original = spaced.group(0).strip()
        compact = re.sub(r"\s+", "", original)
        if len(compact) >= 2:
            suggestions.append(_codex_discovery_suggestion(segment, original, compact.upper() if len(compact) <= 4 else compact, "proper_noun", 0.72, "ASR 出现字母间隔英文/缩写，建议作为工具名或缩写候选复核。", evidence))
    support_phrase = _best_candidate_discovery_support_phrase(evidence)
    original_span = _candidate_discovery_original_span(text)
    if support_phrase and original_span and (
        "deictic_or_screen_reference" in reasons
        or "fragmented_or_semantically_weak_phrase" in reasons
        or "support_evidence_semantically_differs" in reasons
    ):
        kind = _candidate_discovery_kind(text, support_phrase)
        original_for_suggestion = _candidate_discovery_original_for_kind(text, original_span, support_phrase, kind)
        candidate_for_suggestion = _candidate_discovery_candidate_for_kind(original_for_suggestion, support_phrase, kind)
        if kind not in {"number", "action"}:
            bounded_replacement = _bounded_discovery_replacement(text, support_phrase)
            if bounded_replacement is not None:
                original_for_suggestion, candidate_for_suggestion = bounded_replacement
        decisive_number = _candidate_discovery_decisive_number(text, original_for_suggestion, candidate_for_suggestion, kind, evidence)
        decisive_action = _candidate_discovery_decisive_action(text, original_for_suggestion, candidate_for_suggestion, kind, evidence)
        auto_safe = _candidate_discovery_auto_safe(text, original_for_suggestion, candidate_for_suggestion, kind, evidence)
        confidence = 0.96 if decisive_number else (0.94 if decisive_action else (0.93 if auto_safe else (0.78 if "support_evidence_semantically_differs" in reasons else 0.68)))
        reason = "数字/金额/年份疑似错词有 OCR/字幕/人工等强证据直接支撑，可进入高风险数字校验。" if decisive_number else ("动作/步骤疑似错词有视觉/连续片段/OCR/人工等强证据直接支撑，可进入高风险动作校验。" if decisive_action else ("低风险 ASR/字幕疑似错词有 OCR/视觉/字幕/打标强证据支持，可进入自动语义纠错校验。" if auto_safe else "ASR/字幕是低信息指代或碎片表达，屏幕/OCR/视觉/打标证据提供了更具体的候选语义。"))
        suggestions.append(_codex_discovery_suggestion(segment, original_for_suggestion, candidate_for_suggestion, kind, confidence, reason, evidence, needs_human_review=not (auto_safe or decisive_number or decisive_action)))
    boundary_kind = ""
    if "needs_segment_boundary_review" in reasons:
        boundary_kind = "segment_boundary"
    elif "needs_punctuation_review" in reasons or "long_unpunctuated_asr" in reasons:
        boundary_kind = "punctuation"
    if boundary_kind and len(text) >= 45:
        label = "【待断句复核】" if boundary_kind == "segment_boundary" else "【待标点复核】"
        reason = "长段 ASR 缺少可靠分段，建议进入人工/LLM 断句复核；本地 Codex 不直接给最终改写。" if boundary_kind == "segment_boundary" else "长句 ASR 缺少可靠标点，建议进入人工/LLM 标点复核；本地 Codex 不直接给最终改写。"
        suggestions.append(_codex_discovery_suggestion(segment, text[: min(len(text), 120)], label, boundary_kind, 0.42, reason, evidence, needs_human_review=True))
    return suggestions





def _candidate_discovery_original_for_kind(text: str, original_span: str, support_phrase: str, kind: str) -> str:
    if kind == "number":
        support_markers = set(_fact_value_markers(support_phrase))
        markers = _fact_value_markers(text)
        for marker in markers:
            if marker not in support_markers:
                return marker
        if markers:
            return markers[0]
    if kind == "action":
        support_actions = set(_action_markers(support_phrase))
        actions = _action_markers(text)
        for action in actions:
            if action not in support_actions:
                return action
        if actions:
            return actions[0]
    return original_span


def _candidate_discovery_candidate_for_kind(original: str, support_phrase: str, kind: str) -> str:
    if kind == "number":
        original_markers = set(_fact_value_markers(original))
        for marker in _fact_value_markers(support_phrase):
            if marker not in original_markers:
                return marker
    if kind == "action":
        original_actions = set(_action_markers(original))
        for action in _action_markers(support_phrase):
            if action not in original_actions:
                return action
    return support_phrase


def _bounded_discovery_replacement(text: str, support_phrase: str) -> tuple[str, str] | None:
    """Reduce a cross-source suggestion to the shared local target span.

    Prefix/suffix context such as a relative date remains outside the editable
    span, so evidence about a concept cannot silently remove another semantic
    anchor from the cue.
    """

    original = str(text or "").strip()
    suggested = str(support_phrase or "").strip()
    if not original or not suggested:
        return None
    blocks = [
        block
        for block in difflib.SequenceMatcher(None, original, suggested, autojunk=False).get_matching_blocks()
        if block.size >= 2
    ]
    if not blocks:
        return None
    original_start = min(block.a for block in blocks)
    original_end = max(block.a + block.size for block in blocks)
    suggested_start = min(block.b for block in blocks)
    suggested_end = max(block.b + block.size for block in blocks)
    original_span = original[original_start:original_end].strip(" ，。,.!?！？；;：:")
    suggested_span = suggested[suggested_start:suggested_end].strip(" ，。,.!?！？；;：:")
    if (
        not original_span
        or not suggested_span
        or _normalize_compact(original_span) == _normalize_compact(suggested_span)
        or len(original_span) > 80
        or len(suggested_span) > 80
    ):
        return None
    return original_span, suggested_span

def _candidate_discovery_decisive_number(text: str, original: str, support_phrase: str, kind: str, evidence: list[dict[str, Any]]) -> bool:
    if kind != "number":
        return False
    original_markers = set(_fact_value_markers(original))
    support_markers = set(_fact_value_markers(support_phrase))
    if not support_markers or support_markers == original_markers:
        return False
    strong_types = {"ocr", "structured_visual", "platform_subtitle", "embedded_subtitle", "human_note"}
    for item in evidence:
        source_type = str(item.get("source_type") or "")
        if source_type not in strong_types:
            continue
        evidence_markers = set(_fact_value_markers(str(item.get("text") or "")))
        if support_markers & evidence_markers:
            return True
    return False


def _candidate_discovery_decisive_action(text: str, original: str, support_phrase: str, kind: str, evidence: list[dict[str, Any]]) -> bool:
    if kind != "action":
        return False
    original_actions = set(_action_markers(original))
    support_actions = set(_action_markers(support_phrase))
    if not support_actions or support_actions == original_actions:
        return False
    strong_types = {"visual_understanding", "temporal_visual", "structured_visual", "ocr", "human_note"}
    for item in evidence:
        source_type = str(item.get("source_type") or "")
        if source_type not in strong_types:
            continue
        evidence_actions = set(_action_markers(str(item.get("text") or "")))
        if support_actions & evidence_actions:
            return True
    return False

def _candidate_discovery_auto_safe(text: str, original: str, support_phrase: str, kind: str, evidence: list[dict[str, Any]]) -> bool:
    if kind in HIGH_RISK_TYPES or kind == "action":
        return False
    if kind not in {"proper_noun", "term", "concept", "ordinary_word"}:
        return False
    if not original.strip() or not support_phrase.strip():
        return False
    if len(original.strip()) > 80 or len(support_phrase.strip()) > 80:
        return False
    combined = f"{text} {support_phrase}"
    if _fact_value_markers(combined) or ACTION_HINT_RE.search(combined):
        return False
    strong_types = {"ocr", "structured_visual", "visual_understanding", "temporal_visual", "tagger", "platform_subtitle", "embedded_subtitle", "human_note"}
    support_key = _normalize_compact(support_phrase)
    if not support_key:
        return False
    for item in evidence:
        source_type = str(item.get("source_type") or "")
        if source_type not in strong_types:
            continue
        evidence_text = str(item.get("text") or "")
        if support_key and support_key in _normalize_compact(evidence_text):
            return True
    return False

def _codex_discovery_suggestion(segment: dict[str, Any], original: str, candidate: str, kind: str, confidence: float, reason: str, evidence: list[dict[str, Any]], *, needs_human_review: bool = True) -> dict[str, Any]:
    evidence_summary = _candidate_discovery_evidence_summary(evidence)
    return {
        "source_segment_index": segment.get("source_segment_index"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "time_range": segment.get("time_range"),
        "correction_type": kind if kind in VALID_TYPES else "ordinary_word",
        "original_text": original,
        "candidate_text": candidate,
        "reason": reason,
        "confidence": round(float(confidence), 3),
        "evidence_summary": evidence_summary,
        "needs_human_review": bool(needs_human_review),
    }


def _candidate_discovery_original_span(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= 36:
        return stripped
    match = DEICTIC_OR_SCREEN_REF_RE.search(stripped)
    if match:
        start = max(0, match.start() - 8)
        end = min(len(stripped), match.end() + 16)
        return stripped[start:end].strip(" ，。,.!?！？；;：:")
    return stripped[:36].strip(" ，。,.!?！？；;：:")


def _best_candidate_discovery_support_phrase(evidence: list[dict[str, Any]]) -> str:
    source_rank = {
        "ocr": 60,
        "structured_visual": 60,
        "platform_subtitle": 50,
        "embedded_subtitle": 50,
        "visual_understanding": 45,
        "temporal_visual": 45,
        "tagger": 30,
        "page_metadata": 10,
    }
    candidates: list[tuple[int, str]] = []
    for item in evidence:
        source_type = str(item.get("source_type") or "")
        text = _clean_candidate_discovery_support_text(str(item.get("text") or ""))
        if not text:
            continue
        rank = source_rank.get(source_type, 1)
        for phrase in _candidate_discovery_phrases(text):
            if phrase in GENERIC_SUPPORT_PHRASES:
                continue
            score = rank * 100 + min(20, len(phrase))
            candidates.append((score, phrase))
    if not candidates:
        return ""
    candidates.sort(key=lambda row: (-row[0], len(row[1])))
    return candidates[0][1]


def _clean_candidate_discovery_support_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\[\]{}<>#*_`|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _candidate_discovery_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for chunk in re.split(r"[，。！？；：、,.!?;:\n\r\t]+", text):
        chunk = chunk.strip(" -—:：；;，,。.")
        if 4 <= len(chunk) <= 32 and chunk.lower() not in SUPPORT_STOP_TOKENS:
            phrases.append(chunk)
        for sub in re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\- ]{3,28}", chunk):
            sub = sub.strip()
            if 4 <= len(sub) <= 32:
                phrases.append(sub)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(phrase)
    return deduped


def _candidate_discovery_kind(text: str, support_phrase: str) -> str:
    combined = text + " " + support_phrase
    if _fact_value_markers(combined):
        return "number"
    if ACTION_HINT_RE.search(combined):
        return "action"
    if ASCII_TOKEN_RE.search(support_phrase) or ASCII_PHRASE_RE.search(support_phrase):
        return "proper_noun"
    return "concept"


def _candidate_discovery_evidence_summary(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence[:4]:
        source = str(item.get("source_type") or "evidence")
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(f"{source}: {text[:120]}")
    return " | ".join(parts)


def _render_codex_candidate_suggestions_markdown(payload: dict[str, Any]) -> str:
    return "\n".join([
        "# transcript-semantic-candidate-suggestions.codex.md",
        "",
        "本文件由本地 Codex 替代流程生成，只包含候选建议。必须先 import-transcript-semantic-candidate-suggestions，再走 Codex/LLM/人工仲裁、validate 和 closure。",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ])
def _normalise_candidate_suggestions_payload(payload: dict[str, Any], *, discovery_pack: dict[str, Any]) -> dict[str, Any]:
    suggestions = payload.get("suggestions") if isinstance(payload.get("suggestions"), list) else payload.get("candidates")
    return {
        "schema": CANDIDATE_SUGGESTIONS_SCHEMA,
        "source": str(payload.get("source") or "text_llm_candidate_discovery"),
        "bundle_dir": str(discovery_pack.get("bundle_dir") or ""),
        "discovery_pack_json": str((discovery_pack.get("artifacts") or {}).get("candidate_discovery_pack_json") or ""),
        "suggestions": [row for row in (suggestions or []) if isinstance(row, dict)],
        "operator_boundary": {
            "execute_may_call_text_llm": True,
            "suggestions_only": True,
            "does_not_modify_raw_sources": True,
            "does_not_modify_correction_pack": True,
            "import_and_validation_required_before_closure": True,
            "api_key_not_persisted": True,
        },
        "updated_at": now_iso(),
    }


def _render_candidate_suggestions_markdown(payload: dict[str, Any]) -> str:
    return "\n".join([
        "# transcript-semantic-candidate-suggestions.llm.md",
        "",
        "本文件由 text LLM provider 生成，只包含候选建议。必须先 import-transcript-semantic-candidate-suggestions，再走 validate / closure。",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ])
def _render_candidate_discovery_prompt(
    *,
    root: Path,
    pack: dict[str, Any],
    segments: list[dict[str, Any]],
    template: dict[str, Any],
    scan_contract: dict[str, Any] | None = None,
) -> str:
    payload = {
        "bundle_dir": str(root),
        "title": pack.get("title") or root.name,
        "task": "Find additional suspicious ASR/subtitle words that rule-based candidate extraction may have missed. Do not decide or apply corrections.",
        "rules": [
            "Return candidate suggestions only; do not rewrite the transcript.",
            "original_text must be an exact short span from the provided segment text.",
            "Previous/next cues are read-only context; candidate_text may replace only original_text in the target source_segment_index.",
            "Use candidate_text only when cross-evidence or context strongly suggests a better form.",
            "Actively inspect ordinary Chinese wording, not only numbers or actions: use ordinary_homophone for a likely same-pronunciation ASR word error and collocation_anomaly for a contextually abnormal word combination.",
            "For ordinary_homophone or collocation_anomaly, cite the exact local context or independent evidence, keep needs_human_review=true unless two independent direct sources literally support candidate_text, and never use a language-model guess as proof.",
            "Numbers, money, dates, percentages, prices, claims, names, and tool names are high risk; keep needs_human_review=true unless evidence is decisive.",
            "If a segment has no real suspicious span, do not invent one.",
            "Output a JSON object with schema and suggestions only.",
        ],
        "output_schema": template,
        "scan_contract": scan_contract or {},
        "segments": segments,
    }
    return "# 转写语义纠错候选发现任务\n\n请只返回 JSON，不要输出解释性正文。此任务只发现候选，不直接纠错。\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n"


def _merge_imported_discovery_candidates(
    root: Path,
    candidates: list[dict[str, Any]],
    cues: list[Any],
    timeline: list[Any],
    *,
    sidecar_sources: list[dict[str, Any]],
    metadata_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep imported LLM/Codex discovery candidates across pack rebuilds."""
    sidecar = _read_optional_json(root / "transcript-semantic-candidate-suggestions-imported.json")
    imported_rows = sidecar.get("imported_candidates") if isinstance(sidecar, dict) else []
    if not isinstance(imported_rows, list) or not imported_rows:
        return candidates
    merged = [dict(row) for row in candidates if isinstance(row, dict)]
    seen_keys = {
        (
            str(row.get("original_text") or "").lower(),
            str(row.get("candidate_text") or row.get("suggested_text") or "").lower(),
            str(row.get("correction_type") or "ordinary_word"),
        )
        for row in merged
    }
    used_ids = {str(row.get("candidate_id") or "") for row in merged if str(row.get("candidate_id") or "")}
    next_index = len(merged) + 1
    for row in imported_rows:
        candidate: dict[str, Any] | None = None
        if isinstance(row, dict) and row.get("candidate") and isinstance(row.get("candidate"), dict):
            candidate = dict(row.get("candidate") or {})
        elif isinstance(row, dict):
            candidate = dict(row)
        if not candidate:
            continue
        persisted_candidate_id = str(candidate.get("candidate_id") or "")
        discovered_by = str(candidate.get("discovered_by") or "codex_or_llm_candidate_discovery")
        refresh_input = dict(candidate)
        if "confidence" not in refresh_input and "discovery_confidence" in refresh_input:
            refresh_input["confidence"] = refresh_input.get("discovery_confidence")
        refreshed, _ = _candidate_from_discovery_suggestion(
            refresh_input,
            cues,
            timeline,
            sidecar_sources=sidecar_sources,
            metadata_evidence=metadata_evidence,
        )
        if refreshed is None:
            continue
        candidate = refreshed
        if persisted_candidate_id:
            candidate["candidate_id"] = persisted_candidate_id
        candidate["discovered_by"] = discovered_by
        key = (
            str(candidate.get("original_text") or "").lower(),
            str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").lower(),
            str(candidate.get("correction_type") or "ordinary_word"),
        )
        if not key[0] or key in seen_keys:
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in used_ids:
            while f"semcorr-{next_index:04d}" in used_ids:
                next_index += 1
            candidate_id = f"semcorr-{next_index:04d}"
            candidate["candidate_id"] = candidate_id
        used_ids.add(candidate_id)
        seen_keys.add(key)
        candidate.setdefault("discovered_by", "codex_or_llm_candidate_discovery")
        merged.append(candidate)
    return merged
def _candidate_from_discovery_suggestion(
    suggestion: dict[str, Any],
    cues: list[Any],
    timeline: list[Any],
    *,
    sidecar_sources: list[dict[str, Any]],
    metadata_evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    original = str(suggestion.get("original_text") or suggestion.get("span") or "").strip()
    if not original:
        return None, "missing_original_text"
    kind = str(suggestion.get("correction_type") or "ordinary_word").strip()
    if kind not in VALID_TYPES:
        kind = "ordinary_word"
    candidate_text = str(suggestion.get("candidate_text") or suggestion.get("suggested_text") or suggestion.get("corrected_text") or "").strip()
    cue_idx = _resolve_discovery_cue_index(suggestion, cues, original)
    if cue_idx is None:
        return None, "source_segment_not_found"
    cue = cues[cue_idx]
    full_text = str(getattr(cue, "text", "") or "").strip()
    if original not in full_text:
        return None, "original_text_not_in_segment"
    start = _float(suggestion.get("start"), _float(getattr(cue, "start", 0.0)))
    end = _float(suggestion.get("end"), _float(getattr(cue, "end", start), start))
    tl_item, timeline_alignment = _timeline_match_for_cue(
        timeline,
        cue,
        start,
        end,
    )
    evidence = _dedupe_evidence([
        *_evidence_for_cue(
            cue,
            tl_item,
            segment_index=cue_idx,
            timeline_alignment=timeline_alignment,
        ),
        *_sidecar_evidence_for_cue(cue, sidecar_sources),
        *metadata_evidence,
        {
            "evidence_id": "candidate_discovery_suggestion",
            "source_type": "candidate_discovery_suggestion",
            "start": start,
            "end": end,
            "text": str(suggestion.get("evidence_summary") or suggestion.get("reason") or "")[:1200],
        },
    ])
    risk = "high" if kind in HIGH_RISK_TYPES else ("medium" if bool(suggestion.get("needs_human_review", True)) else "low")
    candidate = _candidate(
        cue_idx,
        start,
        end,
        full_text,
        original,
        kind,
        risk,
        evidence,
        reason=str(suggestion.get("reason") or "llm_discovered_candidate"),
        candidate_text=candidate_text,
        has_conflict=bool(candidate_text and _normalize_compact(candidate_text) != _normalize_compact(original)),
    )
    candidate["needs_human_review"] = bool(suggestion.get("needs_human_review", candidate.get("needs_human_review", True))) or kind in HIGH_RISK_TYPES
    candidate["discovery_confidence"] = _float(suggestion.get("confidence"), 0.0)
    candidate["discovery_source_segment_index"] = cue_idx
    return candidate, "ok"


def _resolve_discovery_cue_index(suggestion: dict[str, Any], cues: list[Any], original: str) -> int | None:
    for key in ("source_segment_index", "segment_index"):
        value = suggestion.get(key)
        if str(value).lstrip("-").isdigit():
            idx = int(value)
            if 0 <= idx < len(cues):
                return idx
    start = suggestion.get("start")
    end = suggestion.get("end")
    if start is not None or end is not None:
        s = _float(start, -1.0)
        e = _float(end, s)
        for idx, cue in enumerate(cues):
            cue_start = _float(getattr(cue, "start", 0.0))
            cue_end = _float(getattr(cue, "end", cue_start), cue_start)
            if cue_end >= s and cue_start <= e and original in str(getattr(cue, "text", "") or ""):
                return idx
    for idx, cue in enumerate(cues):
        if original in str(getattr(cue, "text", "") or ""):
            return idx
    return None
def _artifact_paths(root: Path) -> dict[str, str]:
    return {"pack_json": str(root / "transcript-semantic-correction-pack.json"), "prompt_markdown": str(root / "transcript-semantic-correction-prompt.md"), "llm_prompt_markdown": str(root / "transcript-semantic-correction-llm-prompt.md"), "candidate_discovery_pack_json": str(root / "transcript-semantic-candidate-discovery-pack.json"), "candidate_discovery_prompt_markdown": str(root / "transcript-semantic-candidate-discovery-prompt.md"), "candidate_discovery_llm_prompt_markdown": str(root / "transcript-semantic-candidate-discovery-llm-prompt.md"), "candidate_suggestions_llm_json": str(root / "transcript-semantic-candidate-suggestions.llm.json"), "candidate_suggestions_llm_markdown": str(root / "transcript-semantic-candidate-suggestions.llm.md"), "candidate_suggestions_codex_markdown": str(root / "transcript-semantic-candidate-suggestions.codex.md"), "candidate_suggestions_import_json": str(root / "transcript-semantic-candidate-suggestions-import.json"), "result_template_json": str(root / "transcript-semantic-correction-result.template.json"), "result_codex_markdown": str(root / "transcript-semantic-correction-result.codex.md"), "result_llm_json": str(root / "transcript-semantic-correction-result.llm.json"), "result_llm_markdown": str(root / "transcript-semantic-correction-result.llm.md"), "validation_json": str(root / "transcript-semantic-correction-validation.json"), "review_json": str(root / "transcript-semantic-correction-review.json"), "review_markdown": str(root / "transcript-semantic-correction-review.md"), "closure_json": str(root / "transcript-semantic-correction-closure.json"), "impact_json": str(root / "transcript-semantic-correction-impact-report.json"), "readable_impact_json": str(root / "transcript-semantic-readable-impact-report.json"), "readable_impact_markdown": str(root / "transcript-semantic-readable-impact-report.md"), "status_json": str(root / "transcript-semantic-correction-status.json"), "corrected_transcript_json": str(root / "source-arbitrated-transcript.json")}


def _write_mcp_args(root: Path, *, limit: int) -> None:
    codex_path = root / "transcript-semantic-correction-result.codex.md"
    write_json(root / "mcp-transcript-semantic-correction-pack.args.json", {"bundle_dir": str(root), "limit": int(limit or 0), "write": True})
    write_json(root / "mcp-transcript-semantic-correction-codex-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "min_confidence": 0.88, "write": True})
    write_json(root / "mcp-transcript-semantic-correction-llm-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "execute": False, "limit": 80, "min_confidence": 0.88, "write": True})
    write_json(root / "mcp-validate-transcript-semantic-correction.args.json", {"bundle_dir": str(root), "input_json": str(codex_path), "min_confidence": 0.88, "write": True})
    write_json(root / "mcp-transcript-semantic-correction-closure.args.json", {"bundle_dir": str(root), "input_json": str(codex_path), "min_confidence": 0.88, "auto_apply": False, "refresh_exports": True, "write": True})
    write_json(root / "mcp-transcript-semantic-correction-impact-report.args.json", {"bundle_dir": str(root), "write": True})
    write_json(root / "mcp-transcript-semantic-readable-impact-report.args.json", {"bundle_dir": str(root), "write": True})
    write_json(root / "mcp-transcript-semantic-correction-status.args.json", {"bundle_dir": str(root), "write": True})


def _render_prompt(result: dict[str, Any]) -> str:
    lines = ["# 转写语义纠错 Codex 审核任务", "", "只判断 pack 中的 candidates，不要自由改写整篇 transcript。", "", "规则：", "- 综合 ASR、OCR/ebook、视觉理解、时间线、打标器和上下文证据。", "- 数字/金额/比例/年份必须特别保守。", "- 低置信或多种可能都合理时，返回 accept=false 或 needs_human_review=true。", "", "## 输出 Schema", "", "```json", json.dumps(_result_template(result), ensure_ascii=False, indent=2), "```", "", "## 候选概览"]
    for row in result.get("candidates", [])[:80]:
        lines.append(f"- `{row.get('candidate_id')}` [{row.get('correction_type')}] {row.get('time_range')}: `{row.get('original_text')}` | reason={row.get('reason')}")
    return "\n".join(lines) + "\n"


def _result_template(result: dict[str, Any]) -> dict[str, Any]:
    first = (result.get("candidates") or [{}])[0]
    return {
        "schema": RESULT_SCHEMA,
        "source": "codex_or_llm_review",
        "decisions": [
            {
                "candidate_id": first.get("candidate_id", "semcorr-0001"),
                "action": "replace | needs_human_review | reject",
                "correction_type": (
                    "term | proper_noun | number | action | concept | "
                    "ordinary_word | ordinary_homophone | collocation_anomaly | negation | condition | exception | "
                    "person_name | organization | date_time | money | "
                    "percentage | punctuation | segment_boundary"
                ),
                "original_text": first.get("original_text", ""),
                "corrected_text": "",
                "confidence": 0.0,
                "rationale": "",
                "evidence_ids": first.get("evidence_ids", []),
                "human_confirmed": False,
                "needs_human_review": True,
            }
        ],
    }



def _render_codex_result_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# transcript-semantic-correction-result.codex.md",
        "",
        "本文件由本地保守 Codex substitute draft 生成。只包含高置信、可程序校验的局部替换；未列出的候选仍需 Codex/LLM/人工复核。",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)

def _render_codex_stub(template: dict[str, Any]) -> str:
    return "\n".join(["# transcript-semantic-correction-result.codex.md", "", "把 Codex/LLM 审核后的 JSON 放在下面代码块中。未确认的候选不要写成 accept=true。", "", "```json", json.dumps(template, ensure_ascii=False, indent=2), "```", ""])


def _render_semantic_review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# 转写语义纠错人工复核", "", f"- Review required: `{len(rows)}`", ""]
    for row in rows:
        lines.append(f"## `{row.get('candidate_id')}` {row.get('time_range') or ''}")
        lines.append(f"- Type: `{row.get('correction_type')}`")
        lines.append(f"- Original: `{row.get('original_text')}`")
        lines.append(f"- Suggested: `{row.get('suggested_text')}`")
        lines.append(f"- Reject reasons: `{', '.join(row.get('reject_reasons') or [])}`")
        if row.get("evidence_excerpt"):
            lines.append(f"- Evidence: {row.get('evidence_excerpt')}")
        lines.append("")
    return "\n".join(lines) + "\n"

def _render_validation_markdown(result: dict[str, Any]) -> str:
    lines = ["# 转写语义纠错校验", "", f"- Status: `{result.get('status')}`", f"- Accepted: `{result.get('accepted_decision_count')}`", f"- Arbitrated no change: `{result.get('arbitrated_no_change_count')}`", f"- Rejected: `{result.get('rejected_decision_count')}`", ""]
    for row in result.get("decisions", []):
        mark = "OK" if row.get("accepted") else ("NO_CHANGE" if row.get("arbitrated_no_change") else "REJECT")
        lines.append(f"- {mark} `{row.get('candidate_id')}` `{row.get('original_text')}` -> `{row.get('corrected_text')}` reasons={row.get('reject_reasons')}")
    return "\n".join(lines) + "\n"


def _render_closure_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 转写语义纠错闭环写入",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Applied corrections: `{result.get('applied_correction_count')}`",
        f"- Changed segments: `{result.get('changed_segment_count')}`",
        f"- Refresh exports requested: `{bool(result.get('refresh_exports_requested'))}`",
        f"- Refresh exports status: `{result.get('refresh_exports_status', 'not_requested')}`",
        "",
    ]
    refresh = result.get("refresh_exports") if isinstance(result.get("refresh_exports"), dict) else {}
    if refresh:
        lines.extend([
            "## 导出刷新",
            "",
            f"- Full transcript: `{refresh.get('full_transcript_path', '')}`",
            f"- Full body: `{refresh.get('full_body_path', '')}`",
            f"- Smart summary: `{refresh.get('smart_summary_path', '')}`",
            f"- Impact: `{refresh.get('impact_status', '')}`",
            f"- Readable impact: `{refresh.get('readable_impact_status', '')}`",
            f"- Summary impact: `{refresh.get('summary_impact_status', '')}`",
            "",
        ])
    for row in result.get("applied_corrections", []):
        lines.append(f"- `{row.get('candidate_id')}` `{row.get('original_text')}` -> `{row.get('corrected_text')}`")
    return "\n".join(lines) + "\n"
def _render_impact_markdown(result: dict[str, Any]) -> str:
    lines = ["# 转写语义纠错影响报告", "", f"- Status: `{result.get('status')}`", f"- Accepted decisions: `{result.get('accepted_decision_count')}`", f"- Final residual errors: `{result.get('final_residual_error_total')}`", ""]
    for row in result.get("corrections", []):
        lines.append(f"- `{row.get('candidate_id')}` residual={row.get('final_residual_count')} corrected_hits={row.get('final_corrected_count')}: `{row.get('original_text')}` -> `{row.get('corrected_text')}`")
    return "\n".join(lines) + "\n"


def _render_status_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 转写语义纠错状态",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Next action: `{result.get('next_action_key')}`",
        f"- Candidates: `{result.get('candidate_count')}`",
        f"- Candidate groups: `{result.get('candidate_group_count')}`",
        f"- Accepted decisions: `{result.get('accepted_decision_count')}`",
        f"- Arbitrated no change: `{result.get('arbitrated_no_change_count')}`",
        f"- Review required: `{result.get('review_required_count')}`",
        f"- Final residual errors: `{result.get('final_residual_error_total')}`",
        f"- Readable impact: `{result.get('readable_impact_status')}`",
        f"- Readable residual errors: `{result.get('readable_required_residual_total')}`",
        f"- LLM draft status: `{result.get('llm_draft_status')}`",
        f"- LLM draft next action: `{result.get('llm_draft_next_action')}`",
        f"- LLM draft decisions: `{result.get('llm_draft_decision_count')}`",
        f"- Candidate discovery status: `{result.get('candidate_discovery_status')}`",
        f"- Candidate discovery next action: `{result.get('candidate_discovery_next_action')}`",
        f"- Candidate discovery segments/suggestions/imported: `{result.get('candidate_discovery_segment_count')}` / `{result.get('candidate_discovery_suggestion_count')}` / `{result.get('candidate_discovery_imported_candidate_count')}`",
        "",
    ]
    ui_summary = result.get("ui_summary") if isinstance(result.get("ui_summary"), dict) else {}
    if ui_summary:
        export_chain = ui_summary.get("export_chain") if isinstance(ui_summary.get("export_chain"), dict) else {}
        lines.extend([
            "## UI/队列摘要",
            "",
            f"- UI state: `{ui_summary.get('ui_state')}`",
            f"- Auto candidates / human review candidates: `{ui_summary.get('auto_candidate_count')}` / `{ui_summary.get('human_review_candidate_count')}`",
            f"- Accepted / no-change / rejected / applied: `{ui_summary.get('accepted_decision_count')}` / `{ui_summary.get('arbitrated_no_change_count')}` / `{ui_summary.get('rejected_decision_count')}` / `{ui_summary.get('applied_correction_count')}`",
            f"- Review imported / closed / open: `{ui_summary.get('review_imported_count')}` / `{ui_summary.get('review_closed_count')}` / `{ui_summary.get('review_required_count')}`",
            f"- Readable impact / summary impact: `{export_chain.get('readable_impact_status')}` / `{export_chain.get('summary_impact_status')}`",
            "",
        ])
        for title, values in (("已接受类型", ui_summary.get("accepted_decision_type_counts")), ("已应用类型", ui_summary.get("applied_correction_type_counts")), ("拒绝原因", ui_summary.get("rejected_decision_reason_counts"))):
            if not isinstance(values, dict) or not values:
                continue
            lines.extend([f"### {title}", "", "| 项 | 数量 |", "| --- | ---: |"])
            for name, count in sorted(values.items()):
                lines.append(f"| `{name}` | {count} |")
            lines.append("")
    for title, key in (("候选类型", "candidate_type_counts"), ("风险等级", "risk_level_counts"), ("证据来源", "evidence_source_counts"), ("预检拒绝原因", "validation_rejection_reason_counts")):
        values = result.get(key) if isinstance(result.get(key), dict) else {}
        if not values:
            continue
        lines.extend([f"## {title}", "", "| 项 | 数量 |", "| --- | ---: |"])
        for name, count in sorted(values.items()):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
    source_vote = result.get("source_vote_summary") if isinstance(result.get("source_vote_summary"), dict) else {}
    if source_vote:
        lines.extend([
            "## 来源投票 / 字幕可靠性摘要",
            "",
            f"- Candidates with votes: `{int(source_vote.get('candidate_count_with_votes') or 0)}`",
            f"- Source conflicts: `{int(source_vote.get('source_conflict_count') or 0)}`",
            f"- Needs review by source vote: `{int(source_vote.get('needs_review_by_source_vote_count') or 0)}`",
            f"- Candidate / original / neutral weights: `{int(source_vote.get('candidate_weight_total') or 0)}` / `{int(source_vote.get('original_weight_total') or 0)}` / `{int(source_vote.get('neutral_weight_total') or 0)}`",
            "",
        ])
        for title, key in (("Dominant side", "by_dominant_side"), ("Candidate support source", "by_candidate_support_source"), ("Original support source", "by_original_support_source")):
            values = source_vote.get(key) if isinstance(source_vote.get(key), dict) else {}
            if not values:
                continue
            lines.extend([f"### {title}", "", "| 项 | 数量 |", "| --- | ---: |"])
            for name, count in sorted(values.items()):
                lines.append(f"| `{name}` | {count} |")
            lines.append("")
        conflict_preview = source_vote.get("conflict_preview") if isinstance(source_vote.get("conflict_preview"), list) else []
        if conflict_preview:
            lines.extend(["### Source Conflict Preview", "", "| Candidate | Dominant | Weight | Original | Candidate | Supports candidate | Supports original |", "| --- | --- | --- | --- | --- | --- | --- |"])
            for row in conflict_preview[:10]:
                lines.append(
                    f"| `{row.get('candidate_id')}` | `{row.get('dominant_side')}` | `{row.get('candidate_weight')}/{row.get('original_weight')}` | {str(row.get('original_text') or '').replace('|', '/')} | {str(row.get('candidate_text') or '').replace('|', '/')} | `{', '.join(row.get('supports_candidate') or [])}` | `{', '.join(row.get('supports_original') or [])}` |"
                )
            lines.append("")
    attention = result.get("semantic_attention_preview") if isinstance(result.get("semantic_attention_preview"), list) else []
    if attention:
        lines.extend(["## 语义重点复核队列", "", "| Candidate | 类型 | 分数 | 时间 | 原文 | 建议 | 证据源 |", "| --- | --- | ---: | --- | --- | --- | --- |"])
        for row in attention[:12]:
            sources = ", ".join(str(item) for item in (row.get("evidence_source_types") or [])[:6])
            lines.append(
                f"| `{row.get('candidate_id')}` | `{row.get('correction_type')}` | {int(row.get('priority_score') or 0)} | {str(row.get('time_range') or '').replace('|', '/')} | {str(row.get('original_text') or '').replace('|', '/')} | {str(row.get('suggested_text') or row.get('reason') or '').replace('|', '/')} | `{sources.replace('|', '/')}` |"
            )
        lines.append("")
    group_preview = result.get("candidate_group_preview") if isinstance(result.get("candidate_group_preview"), list) else []
    if group_preview:
        lines.extend(["## 候选分组预览", "", "| Group | Canonical | 类型 | 风险 | 候选 | 变体 | 证据来源 |", "| --- | --- | --- | --- | ---: | --- | --- |"])
        for row in group_preview[:12]:
            variants = ", ".join(str(item) for item in (row.get("variant_texts") or [])[:6])
            sources = ", ".join(str(item) for item in (row.get("evidence_source_types") or [])[:6])
            canonical = str(row.get("canonical_hint") or "").replace("|", "/")
            types = ", ".join(str(item) for item in (row.get("correction_types") or [row.get("correction_type")]) if str(item))
            lines.append(
                f"| `{row.get('candidate_group_id')}` | {canonical} | `{types.replace('|', '/')}` | `{row.get('risk_level')}` | {int(row.get('candidate_count') or 0)} | {variants.replace('|', '/')} | `{sources.replace('|', '/')}` |"
            )
        lines.append("")
    chapter_summary = result.get("chapter_risk_summary") if isinstance(result.get("chapter_risk_summary"), list) else []
    if chapter_summary:
        lines.extend(["## 按章节/风险分组", "", "| 章节 | 时间 | 候选 | 待复核 | 风险 | 高风险候选 |", "| --- | --- | ---: | ---: | --- | --- |"])
        for row in chapter_summary[:20]:
            risks = row.get("risk_level_counts") if isinstance(row.get("risk_level_counts"), dict) else {}
            risk_text = ", ".join(f"{key}={value}" for key, value in sorted(risks.items())) or "none"
            high_ids = ", ".join(str(item) for item in (row.get("high_risk_candidate_ids") or [])[:8])
            chapter = f"{row.get('chapter_index')} {row.get('chapter_title') or ''}".strip()
            lines.append(f"| `{chapter.replace('|', '/')}` | {str(row.get('chapter_time_range') or '').replace('|', '/')} | {int(row.get('candidate_count') or 0)} | {int(row.get('review_required_count') or 0)} | `{risk_text.replace('|', '/')}` | `{high_ids.replace('|', '/')}` |")
        lines.append("")
    preview = result.get("review_required_preview") if isinstance(result.get("review_required_preview"), list) else []
    if preview:
        lines.extend(["## 待人工复核样例", "", "| Candidate | 类型 | 时间 | 原文 | 建议/原因 |", "| --- | --- | --- | --- | --- |"])
        for row in preview[:8]:
            reasons = ", ".join(str(item) for item in row.get("reject_reasons") or [])
            suggestion = str(row.get("suggested_text") or reasons or "")
            lines.append(f"| `{row.get('candidate_id')}` | `{row.get('correction_type')}` | {row.get('time_range')} | {str(row.get('original_text') or '').replace('|', '/')} | {suggestion.replace('|', '/')} |")
        lines.append("")
    lines.append("## Commands")
    lines.append("")
    for key, command in result.get("commands", {}).items():
        lines.append(f"- {key}: `{command}`")
    return "\n".join(lines) + "\n"


def _render_corrected_markdown(payload: dict[str, Any]) -> str:
    segments = [row for row in payload.get("segments", []) if isinstance(row, dict)]
    labels = speaker_label_map(segments)
    lines = ["# Source-Arbitrated Transcript", "", f"- Source: `{payload.get('source')}`", f"- Updated: `{payload.get('updated_at')}`", ""]
    for row in segments:
        lines.append(f"## {format_timestamp(_float(row.get('start')))} - {format_timestamp(_float(row.get('end')))}")
        speaker = speaker_display_name(row, labels)
        if speaker:
            lines.append(f"**{speaker}**")
        lines.append(str(row.get("text") or ""))
        lines.append("")
    return "\n".join(lines)


def _render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    labels = speaker_label_map(segments)
    for idx, row in enumerate(segments, start=1):
        start = format_timestamp(_float(row.get("start"))).replace(".", ",")
        end = format_timestamp(_float(row.get("end"))).replace(".", ",")
        speaker = speaker_display_name(row, labels)
        text = f"{speaker}：{row.get('text', '')}" if speaker else str(row.get("text") or "")
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def _status_from_artifacts(pack: Any, validation: Any, closure: Any, impact: Any, readable_impact: Any | None = None, summary_impact: Any | None = None) -> tuple[str, str]:
    if not isinstance(pack, dict) or not pack:
        return "missing_pack", "build_pack"
    if int(pack.get("candidate_count") or 0) == 0:
        return "no_candidates", "none"
    if not isinstance(validation, dict) or not validation:
        return "needs_llm_or_codex_review", "run_llm_draft_preview"
    if int(validation.get("accepted_decision_count") or 0) == 0:
        if int(validation.get("arbitrated_no_change_count") or 0) > 0 and int(validation.get("rejected_decision_count") or 0) == 0:
            return "arbitrated_no_change", "none"
        return "needs_human_review_or_new_result", "review_candidates"
    if _closure_needs_rerun_after_validation(validation, closure):
        return "needs_closure", "run_closure"
    if not isinstance(impact, dict) or not impact:
        return "needs_impact_report", "run_impact"
    if _artifact_is_older_than(impact, closure):
        return "needs_impact_report", "run_impact"
    if impact.get("status") == "passed":
        if not isinstance(readable_impact, dict) or not readable_impact:
            return "needs_readable_impact_report", "run_readable_impact"
        if _artifact_is_older_than(readable_impact, impact):
            return "needs_readable_impact_report", "run_readable_impact"
        if readable_impact.get("status") == "passed":
            if not isinstance(summary_impact, dict) or not summary_impact:
                return "needs_summary_impact_report", "run_summary_impact"
            if _artifact_is_older_than(summary_impact, readable_impact):
                return "needs_summary_impact_report", "run_summary_impact"
            if summary_impact.get("status") in {"passed", "no_accepted_decisions", "no_evaluable_replacements"}:
                return "impact_passed", "none"
            return "summary_impact_needs_fix", "refresh_summary_or_review"
        return "readable_impact_needs_fix", "refresh_exports_or_review"
    return "impact_needs_fix", "refresh_exports_or_review"


def _closure_needs_rerun_after_validation(validation: Any, closure: Any) -> bool:
    if not isinstance(closure, dict) or not closure:
        return True
    accepted_count = int(validation.get("accepted_decision_count") or 0) if isinstance(validation, dict) else 0
    if accepted_count <= 0:
        return False
    closure_status = str(closure.get("status") or "")
    if closure_status in {"completed", "completed_no_text_changes"}:
        return _artifact_is_older_than(closure, validation)
    if closure_status in {"", "missing", "no_safe_decisions", "no_matching_segments"}:
        return True
    if not bool(closure.get("ok")):
        return True
    return _artifact_is_older_than(closure, validation)


def _artifact_is_older_than(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_time = str(left.get("updated_at") or "")
    right_time = str(right.get("updated_at") or "")
    return bool(left_time and right_time and left_time < right_time)

def _status_commands(root: Path) -> dict[str, str]:
    q = f"'{root}'"
    codex = root / "transcript-semantic-correction-result.codex.md"
    llm = root / "transcript-semantic-correction-result.llm.md"
    return {
        "pack": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-pack {q}",
        "codex_draft": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-codex-draft {q}",
        "llm_draft_preview": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --limit 80",
        "run_llm_draft_preview": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --limit 80",
        "candidate_discovery": f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {q} --limit 40",
        "run_candidate_discovery": f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {q} --limit 40",
        "candidate_discovery_llm_preview": f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --limit 40",
        "run_candidate_discovery_llm_preview": f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --limit 40",
        "execute_candidate_discovery_llm_or_use_codex": f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --execute --provider-config PATH_TO_PROVIDER_CONFIG_JSON  # or fill transcript-semantic-candidate-suggestions.codex.md",
        "import_candidate_suggestions": f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-candidate-suggestions {q} --input-json '{root / 'transcript-semantic-candidate-suggestions.codex.md'}'",
        "execute_llm_or_use_codex": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --execute --provider-config PATH_TO_PROVIDER_CONFIG_JSON  # or run codex_draft",
        "validate_llm_result": f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {q} --input-json '{llm}'",
        "retry_llm_or_manual_review": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --limit 80",
        "validate_result": f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {q} --input-json '{codex}'",
        "validate": f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {q} --input-json '{codex}'",
        "closure": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure {q} --input-json '{codex}' --refresh-exports",
        "impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}",
        "run_impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}",
        "readable_impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-readable-impact-report {q}",
        "run_readable_impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-readable-impact-report {q}",
        "summary_impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}",
        "run_summary_impact": f".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}",
        "run_transcript_quality_gate": f".\\scripts\\video-knowledge.ps1 transcript-quality-gate {q}",
        "refresh_summary_or_review": f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}; .\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}",
        "status": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status {q}",
    }

def _impact_next_actions(status: str) -> list[str]:
    if status == "passed":
        return []
    if status == "no_accepted_decisions":
        return ["Run transcript-semantic-correction-pack, review candidates with Codex/LLM, then validate and close accepted corrections."]
    return ["Regenerate full-transcript and smart-summary from corrected transcript, then rerun transcript-semantic-correction-impact-report."]


def _evidence_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    for row in candidates:
        by_type[str(row.get("correction_type"))] = by_type.get(str(row.get("correction_type")), 0) + 1
        by_risk[str(row.get("risk_level"))] = by_risk.get(str(row.get("risk_level")), 0) + 1
    return {"by_type": by_type, "by_risk": by_risk}


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ("text", "summary", "description", "visual_text", "content"):
            if value.get(key):
                parts.append(str(value.get(key)))
        return " ".join(parts).strip()
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value).strip()
    return str(value).strip()


def _visual_conflict_text(text: str, visual_text: str) -> str:
    visual_text = _strip_technical_visual_artifacts(visual_text)
    if not visual_text:
        return ""
    compact_text = _normalize_compact(text)
    text_values = set(_fact_value_markers(text))
    text_numeric_keys = set(number_evidence_map(text))
    for value in _fact_value_markers(visual_text):
        support_numeric_keys = set(number_evidence_map(value))
        if support_numeric_keys and support_numeric_keys.issubset(text_numeric_keys):
            continue
        if value and value not in text_values and _normalize_compact(value) not in compact_text:
            return value
    tokens = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", visual_text):
        tok_norm = _normalize_compact(tok)
        if tok.lower() in SUPPORT_STOP_TOKENS:
            continue
        if tok_norm in compact_text:
            # For tool/proper names, compact equality can still hide an ASR error:
            # "browser base" and "Browserbase" sound alike but should arbitrate to
            # the source spelling when OCR/subtitle/tagger evidence has it.
            if re.search(r"[A-Za-z]", tok) and tok not in text:
                return tok
            continue
        tokens.append(tok)
    return tokens[0] if tokens else ""


def _strip_technical_visual_artifacts(value: str) -> str:
    """Remove transport/file tokens that must never become semantic evidence."""

    text = str(value or "")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(
        r"(?i)(?:[A-Za-z]:[\\/]|https?://|(?:^|\s)(?:assets?|frames?)[\\/])\S+",
        " ",
        text,
    )
    text = re.sub(r"(?i)\b(?:img|frame)[-_.]?\d+(?:\.(?:bmp|gif|jpe?g|png|webp))?\b", " ", text)
    text = re.sub(r"\b[a-fA-F0-9]{40,64}\b", " ", text)
    text = re.sub(r"(?i)\b[a-z0-9_.-]+\.v\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _support_token_not_in_text(text: str, support_text: str) -> str:
    return _visual_conflict_text(text, support_text)


def _support_concept_phrase_not_in_text(text: str, support_text: str) -> str:
    if not support_text:
        return ""
    compact_text = _normalize_compact(text)
    candidates: list[str] = []
    for phrase in re.findall(r"[\u4e00-\u9fff]{4,18}", support_text):
        phrase = _trim_support_phrase(phrase)
        if _support_phrase_usable(phrase, compact_text):
            candidates.append(phrase)
    for phrase in _ascii_phrases_any_text(support_text):
        if _support_phrase_usable(phrase, compact_text):
            candidates.append(phrase)
    if not candidates:
        token = _support_token_not_in_text(text, support_text)
        return token if _support_phrase_usable(token, compact_text) else ""
    candidates.sort(key=lambda item: (len(item), item), reverse=True)
    return candidates[0]


def _trim_support_phrase(phrase: str) -> str:
    value = str(phrase or "").strip(" ，。！？；：、,.!?;:()（）[]【】")
    for prefix in ("画面显示", "讲师演示", "画面出现", "标签", "标题"):
        if value.startswith(prefix) and len(value) > len(prefix) + 2:
            value = value[len(prefix):].strip(" ：:，,。 ")
    return value


def _support_phrase_usable(phrase: str, compact_text: str) -> bool:
    value = str(phrase or "").strip()
    if len(value) < 4:
        return False
    if value in GENERIC_SUPPORT_PHRASES:
        return False
    if _normalize_compact(value) in compact_text:
        return False
    if NUMBER_RE.fullmatch(value):
        return False
    if FILLER_RE.search(value):
        return False
    return True


def _looks_deictic_or_low_information(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if DEICTIC_OR_SCREEN_REF_RE.search(value):
        return True
    compact = re.sub(r"\s+", "", value)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", compact)
    unique_chars = set(cjk_chars)
    return len(cjk_chars) >= 8 and len(unique_chars) <= max(4, len(cjk_chars) // 3)


def _suspicious_span(text: str, support_token: str) -> str:
    if not support_token:
        return _short_span(text)
    token = str(support_token).strip()
    if NUMBER_RE.fullmatch(token) or CHINESE_FACT_VALUE_RE.fullmatch(token):
        markers = _fact_value_markers(text)
        if markers:
            return markers[0]
    if re.search(r"[A-Za-z]", support_token):
        spaced = ODD_SPACING_RE.search(text)
        if spaced:
            return spaced.group(0)
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
        if words:
            return " ".join(words[: min(4, len(words))])
    return _short_span(text)


def _infer_correction_type(token: str) -> str:
    if NUMBER_RE.fullmatch(token.strip()):
        return "number"
    if re.search(r"[A-Za-z]", token):
        return "proper_noun"
    return "term"


def _normalize_compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _dedupe_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        deduped.append(row)
    return deduped



def _ordinary_subtitle_diff_candidate(text: str, sidecar_text: str) -> tuple[str, str] | None:
    return _ordinary_support_diff_candidate(text, sidecar_text)


def _ordinary_support_diff_candidate(text: str, support_text: str) -> tuple[str, str] | None:
    raw = str(text or "").strip()
    support = str(support_text or "").strip()
    if not raw or not support:
        return None
    if re.search(r"[A-Za-z]", raw + support):
        return None
    if _fact_value_markers(raw) or _fact_value_markers(support):
        return None
    raw_compact = _semantic_diff_compact(raw)
    support_compact = _semantic_diff_compact(support)
    if raw_compact == support_compact:
        return None
    if len(raw_compact) < 8 or len(support_compact) < 8:
        return None
    length_ratio = min(len(raw_compact), len(support_compact)) / max(len(raw_compact), len(support_compact))
    if length_ratio < 0.72:
        window = _best_semantic_diff_window(raw_compact, support_compact)
        if not window:
            return None
        raw_compact = window
    matcher = difflib.SequenceMatcher(a=raw_compact, b=support_compact, autojunk=False)
    ratio = matcher.ratio()
    if ratio < 0.72 or ratio > 0.985:
        return None
    candidates: list[tuple[str, str, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        original = raw_compact[i1:i2].strip()
        suggested = support_compact[j1:j2].strip()
        if not original or not suggested:
            continue
        if len(original) > 8 or len(suggested) > 8:
            continue
        if not re.search(r"[\u4e00-\u9fff]", original + suggested):
            continue
        if original == suggested:
            continue
        if FILLER_RE.search(original) or FILLER_RE.search(suggested):
            continue
        candidates.append((original, suggested, max(len(original), len(suggested))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[2], item[0]))
    original, suggested, _score = candidates[0]
    return original, suggested


def _best_semantic_diff_window(raw_compact: str, support_compact: str) -> str:
    if not raw_compact or not support_compact:
        return ""
    if len(raw_compact) <= len(support_compact):
        return ""
    best_ratio = 0.0
    best_window = ""
    base_len = len(support_compact)
    for window_len in range(max(4, base_len - 2), min(len(raw_compact), base_len + 2) + 1):
        for start in range(0, len(raw_compact) - window_len + 1):
            window = raw_compact[start:start + window_len]
            ratio = difflib.SequenceMatcher(a=window, b=support_compact, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = window
    return best_window if best_ratio >= 0.72 else ""


def _semantic_diff_compact(value: str, *, lowercase: bool = True) -> str:
    text = re.sub(r"[\s，。！？；：、,.!?;:()（）【】\[\]{}<>《》]+", "", str(value or ""))
    for marker in ("\"", "'", "“", "”", "‘", "’", "`"):
        text = text.replace(marker, "")
    return text.lower() if lowercase else text


def _punctuation_or_boundary_kind(text: str, *, start: float, end: float) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 45:
        return ""
    punctuation_count = len(PUNCTUATION_RE.findall(text))
    duration = max(0.0, end - start)
    marker_count = len(BOUNDARY_MARKER_RE.findall(text))
    if punctuation_count >= 2:
        return ""
    if duration >= 25.0 and len(compact) >= 45:
        return "segment_boundary"
    if len(compact) >= 90 and punctuation_count <= 1:
        return "punctuation"
    if marker_count >= 3 and punctuation_count <= 1:
        return "segment_boundary"
    return ""


def _looks_fragmented(text: str) -> bool:
    if len(text) <= 4:
        return False
    if ODD_SPACING_RE.search(text):
        return True
    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text)
    return len(words) >= 5 and len(set(words)) <= max(2, len(words) // 4)


def _short_span(text: str) -> str:
    return text.strip()[:80]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _impact_countable_text(text: str, key: str) -> str:
    """Remove machine artifact paths from human-readable impact counting."""
    if not text:
        return ""
    if key in {"full_transcript", "smart_summary", "smart_summary_codex", "source_arbitrated_transcript_markdown"}:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]*`", "", text)
        text = "\n".join(line for line in text.splitlines() if not re.search(r"[A-Za-z]:[\\\\/]", line))
    return text

def _count_text(text: str, needle: str) -> int:
    if not text or not needle:
        return 0
    return text.count(needle)
