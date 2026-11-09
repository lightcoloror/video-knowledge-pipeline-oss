# 视觉批处理 PowerShell 入口：源码与验证边界

**文档目标**：`docs/vision-batch-entry-source.md`
**对应脚本**：`scripts/run-volcengine-vision-batch.ps1`（开源许可：AGPL-3.0-only）

#### 概述与定位
本脚本为面向火山引擎视觉多模态批处理的 PowerShell 入口包装器，固定声明提供商标识 `$env:LECTURE_VISION_PROVIDER = "volcengine_coding_plan"`。其核心职责是管理环境注入、写入提供商配置、驱动底层预检流程并在调用方显式传入执行开关后调用批处理分析及后续工件刷新。本工具仅为客户端命令行封装，不随附模型权重、测试视频媒体、真实凭据或可用配额。

#### 核心参数与执行语义
1. **输入与环境契约**：
   - `-BundleDir`（必填，位置 0）：目标知识包绝对或相对目录，调用方须提供现有目录；脚本用 `Resolve-Path -LiteralPath` 解析路径，但没有另作目录类型断言。
   - `-EnvFile`：显式指定的环境文件路径，未提供时回退至环境变量 `$env:VKP_PROVIDER_ENV_FILE`，**不存在隐式默认的 auto.env 机制**。脚本信任调用方文件并将其导入当前进程级环境变量；仅在目标变量未设置时映射 `OPENAI_API_KEY -> LLM_API_KEY`、`OPENAI_BASE_URL -> LLM_BASE_URL`、`MODEL_NAME -> LLM_MODEL`。
2. **默认预检模式（非只读）**：
   - 未显式传入 `-Execute` 时，脚本**仍会产生副作用**：导入环境变量、在 `-BundleDir` 下写入 `volcengine-provider-config.json`，并执行 `vision-execution-preflight`。
   - 若预检未通过（`ready_to_execute` 为 false），退出码为 `2`；若预检就绪但未加 `-Execute`，打印预览信息并以退出码 `0` 正常结束。
3. **显式执行模式（`-Execute`）**：
   - 必须显式携带 `-Execute` 方能进入实际推理调用。脚本强制采用预检输出中由后端决定的 `confirm_vision_calls` 与 `confirm_vision_indexes`（注意：索引为时间轴项目业务索引，非数组零基下标）。
   - `-ExecutionActor` 默认为 `"operator"`（可选 `"agent"`）。若指定 `-ExportConsent`，通过 `-LiteralPath` 解析后透传；此类开关仅为 CLI 标志，**不代表独立的隐私合规审查或外部人类授权**。
4. **分析模式与重试控制**：
   - 默认模式：预检传 `--semantic-limit`／`--no-temporal`（有Indexes时加 `--semantic-indexes`）；执行命令为 `run-multimodal-frame-analysis`，使用 `--limit`、可选 `--indexes`。
   - `-Temporal` 模式：预检传 `--temporal-limit`／`--no-semantic`／`--frame-count`（有Indexes时加 `--temporal-indexes`）；执行命令为 `run-temporal-visual-analysis`，使用 `--limit`／`--frame-count`、可选 `--indexes`。
   - 重试参数：`-VisionRetries`（默认 3）与 `-VisionRetryDelaySeconds`（默认 5）透传至底层驱动，包装器自身不执行静默重试。
5. **审计脚本与工件刷新**：
   - 执行前将构造的命令保存为 `$BundleDir/volcengine-last-run-command.ps1`（仅作执行记录审计，包装器不执行该生成文件；`Quote-Arg` 仅为命令行格式化，非注入防护沙箱）。
   - `-NoRefresh` 开关：仅在执行成功后跳过 6 项后续刷新指令（`audit-knowledge-coverage`、`acceptance-check`、`bundle-status-report`、`export-knowledge-note`、`export-task-console`、`refresh-review-html`），**绝非**模拟演练（dry-run）开关。

#### 非执行占位符调用示例
```text
# 1. 预检模式（非只读预检入口；后端实际网络／环境行为本批未验证）
powershell -File scripts/run-volcengine-vision-batch.ps1 -BundleDir "<PATH_TO_SYNTHETIC_BUNDLE_DIR>" -EnvFile "<PATH_TO_SYNTHETIC_ENV_FILE>"

# 2. 显式执行时序视觉批处理（仅示意已有真实有效同意引用的参数形式，不生成或授予同意）
powershell -File scripts/run-volcengine-vision-batch.ps1 -BundleDir "<PATH_TO_SYNTHETIC_BUNDLE_DIR>" -EnvFile "<PATH_TO_SYNTHETIC_ENV_FILE>" -Temporal -FrameCount 8 -Execute -ExportConsent "<PATH_TO_CONSENT_FILE>" -NoRefresh
```


## 本批来源、修改与验证

源自现有通用工作快照；保留原文与来源记录，不冒称固定历史Git原件。AGPL-3.0-only覆盖本批第一方源码；本批不分发后端依赖、媒体、模型、真实EnvFile／账户配置、导出同意或日志。包装器没有嵌入账号值，普通公开API引用不需机械删除。实际未来运行可能在配置／命令记录／后端输出里留下私有路径和数据，不能将它们当作可公开源码。

代码范围仅将仓库父目录的 `Resolve-Path` 改为 `Resolve-Path -LiteralPath`，其余语句和参数／分支／失败处理／刷新顺序保留（发布副本可规范LF）。这是含方括号路径兼容性修补，不是安全沙箱。完整公开后端命令与env／供应商／同意／真实视觉结果／费用未在本批验证；提供商标识和CLI开关不证明当前服务／订阅可用或具有真实导出授权。

合成测试以任务内假后端记录调用与返回预检数据；不读取真实EnvFile／课件帧，不调用实际视觉服务或运行生成的last-run-command文件。本批实际初步验收只执行一次助手启动命令，PowerShell 返回 `AuthorizationManager check failed / SecurityError / UnauthorizedAccess`，退出1、助手尚未启动；5个正向合成用例与1个旧代码负例共6项均未运行，不能写通过数。首错保留，未变更执行策略／权限／TEMP，也未换宿主或方式重试。具体拒绝根因未进一步诊断。预检JSON解析、路径／I/O或底层命令出错时可非零退出，不保证都有结构化结果或自动回滚。


原工作快照7064字节保留，发布副本6901字节，除CRLF转LF与该单处LiteralPath外，全部语句字节相同。Windows PowerShell5.1对入口与私有验收助手的最终非执行解析均0错误；不代表执行通过。助手／原始日志／任务回执不在这份公开增量中；源工具、参数和刷新分支未为制造通过而删减。
