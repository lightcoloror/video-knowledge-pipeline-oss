# 字幕画幅边界检测与安全边距自动修复流水线

## 1. 背景与核心规范

在 9:16 竖屏（1080×1920）口播成片中，字幕如果直接单行渲染过长，两端会被手机端播放器边框、各种平台 UI 按钮（点赞/评论/转发）或屏幕物理边缘裁切。

为此，本流水线确立了代码级硬性门禁规范：
- **画布物理宽度**：`1080 px`
- **安全边距（左右对称）**：`>= 80 px`
- **最大文本物理安全宽度**：`920 px`（`1080 - 80 * 2`）
- **基准字号**：`76 px`（微软雅黑粗体 / PingFang SC Bold）
- **单行安全字数上限**：中文 $\le 11 \sim 12$ 字（12 个汉字在 76px 下物理宽度为 912px，严格 $\le 920\text{px}$）

---

## 2. 工具架构与 CLI 命令

模块路径：[`src/video_knowledge_pipeline/subtitle_boundary_guard.py`](../src/video_knowledge_pipeline/subtitle_boundary_guard.py)
命令行入口：`video-knowledge-subtitle-guard`

### 2.1 边界合规检查（Check Gate）
对给定的 SRT 字幕进行逐行扫描，如果存在任何一行超过 920px 或边距不足 80px，返回非零退出码并输出违规清单：

```bash
# 终端直观查看
python -m video_knowledge_pipeline.subtitle_boundary_guard check path/to/subtitles.srt

# 输出 JSON 格式审计报告供 CI / 质检流水线消费
python -m video_knowledge_pipeline.subtitle_boundary_guard check path/to/subtitles.srt --json report.json
```

### 2.2 自动无损修复（Auto-Fix Gate）
对超宽字幕执行递归安全折行与时间轴比例分配：
1. **标点优先拆分**：在 `，`、`、`、`；`、`。` 处拆分；
2. **安全字符数截断**：无标点时按照 $\le 11$ 字切分为多行；
3. **时长等比例划分**：当切分后的行数 $>2$ 时，自动按字符数占比拆解为连续的独立字幕片段，保持总起止时间完全一致，时间轴无缝衔接。

```bash
python -m video_knowledge_pipeline.subtitle_boundary_guard fix input_overflow.srt --out output_safe.srt
```

---

## 3. Python API 集成示例

```python
from video_knowledge_pipeline.subtitle_boundary_guard import check_srt_file, fix_srt_file

# 1. 运行质量检查门禁
audit_result = check_srt_file("subtitles.srt")
if not audit_result["passed"]:
    print(f"发现违规条目: {audit_result['violating_cues']}")
    # 2. 自动修复并生成安全版
    fix_srt_file("subtitles.srt", "subtitles_repaired.srt")
```
