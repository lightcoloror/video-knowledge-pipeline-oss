# 课件增量 OCR：英文合成验收源码

#### 概述
本脚本提供针对课件增量 OCR 运行时（Incremental OCR Runtime）的离线合成验收逻辑，遵循 AGPL-3.0-only。该脚本通过生成 4 帧结构已知的英文合成图片，校验在一个合成红点、局部文本修改及断点恢复场景下的缓存复用及恢复后两个推理计数归零的检查逻辑。

#### 命令行调用与参数契约
调用必须使用明确的占位符（请勿在未就绪环境下执行缺失权重的命令）：

1. **预检模式（Preflight Mode）**：
   ```text
   python scripts/validate-courseware-incremental-ocr.py --model-policy <PATH_TO_MODEL_POLICY_JSON> --output-root <PATH_TO_NEW_OUTPUT_DIR>
   ```
   - 执行模型策略语法与哈希检查、RapidOCR 探测、系统资源门禁（>=8GiB 内存、<75% CPU）及 LM Studio 互斥检查。
   - 打印 Preflight 状态 JSON。若就绪退出码为 `0`；若存在阻塞项（blockers）退出码为 `2`。

2. **完整执行模式（Execution Mode）**：
   ```text
   python scripts/validate-courseware-incremental-ocr.py --model-policy <PATH_TO_MODEL_POLICY_JSON> --output-root <PATH_TO_NEW_OUTPUT_DIR> --execute
   ```
   - 在预检通过的基础上，生成合成图片并执行增量 OCR 与断点恢复验证。
   - 所有 6 项合成校验通过退出码为 `0`；检查字典出现 false 时退出码为 `2`；未捕获的策略／环境／I/O／运行时异常可能非零退出且没有最终 JSON，不能统称为结构化“检查失败”。

#### 目录与输出文件规范
- 只有 `--execute` 且预检就绪后，脚本才检查 `--output-root` 并要求路径尚不存在，随后用 `Path.absolute()` 处理。若已存在，以字符串错误退出（通常退出1）；不覆盖旧目录。未加 `--execute` 时此参数仍为必填，但不会执行这段目录创建检查。
- 成功执行后将在该目录下生成以下产物：
  ```text
  <output-root>/
  ├── manifest.json                  # 验收标题元数据
  ├── timeline.json                  # 合成时间轴定义
  ├── synthetic-acceptance.json      # 包含 6 项 checks 结果及状态的最终凭据
  └── courseware/
      ├── frames/                    # 4 张合成 PNG 帧（0.png - 3.png）
      ├── pages/                     # OCR 识别产物 JSON
      └── scan 索引与中间缓存
  ```

#### 诚实边界与使用约束
- **输入要求**：调用方必须显式提供符合 `video_knowledge_pipeline.incremental_ocr_models.v1` 架构的 `--model-policy` 文件（指定 Det/Cls/Rec 模型路径与 SHA256、已安装 RapidOCR 版本；代码另检查策略／模型文件大小上限）。本批不分发模型权重与 ONNX 文件，严禁伪造哈希。
- **无沙箱证明**：脚本内 `provider_calls: 0` 字段为内部计数自述，不构成操作系统层面的外联隔离证明。
- **测试局限**：这份源码声明检查合成英文字符串的特定跳过与复用逻辑，本批未运行这些检查，不代表真实复杂课件的识别精度、版面分析能力或生产环境上线就绪。


## 实际依赖与本批验收

四个直接Python模块已在公开父提交中存在，签名能对应本脚本调用；这只是源码／引用证据。预检还需要工作区外部 `ebook_markdown_pipeline.local_env`、所选Python中的RapidOCR与既有本地ONNX、psutil及 `lms` 互斥探测。即使提供 `EBOOK_CONVERTER_RAPIDOCR_PYTHON`，也不移除外部ebook模块的显式导入。所需外部环境、策略、权重及许可证条件由调用方另行准备与核验，本批没有安装、下载或读取实际配置／模型。资源门和互斥逻辑保留，未证明本机满足条件。

脚本没有main导入守卫：导入它会执行顶层import和argparse；`--help`也发生在这些import之后。不要为查看说明而把它当无副作用模块导入。不带 `--execute` 的预检仍可能读取配置／模型文件、计算哈希、启动子进程并探测资源／其他进程；有异常时不保证总能输出状态JSON。

直接依赖元数据含 `pillow>=10,<13`，但没有执行 `ImageFont.load_default(size=32)`，不能据版本范围断言所有版本均兼容。没有分发或验证字体二进制／字形。输出包含中间文件与缓存，失败后可能留下部分产物，不具备全有或全无事务保证，也不提供清理／删除指令。

原3167字节通用工作快照保留，发布副本仅CRLF转LF，语句与AST等价；不冒称固定历史Git原件。仅做不执行的AST语法、完整源码差异／引用／敏感内容／来源核对。Gemini3.8Flash/high承担初步静态验收与说明，Owner限定措辞后最终核对。本批没有导入脚本／后端、运行预检、加载权重、调用OCR或启动服务：6个声明检查，实际执行0个。

5个初始检查为红点前后文字相同、删除行不含SECOND、修改帧文字不同、复用区域数大于0、检测调用数等于4；另检查resume时detector_calls和recognition_calls均为0。文字不同只是不等式，不是修改内容准确识别证明；没有CER阈值，中文／真实复杂课件的效果未验证，不能据此否定后端跨语言能力。两个计数归零也不是所有模型／工作／耗时为零的证明。`passed`和 `provider_calls` 等字段是程序产生的数据，不能替代实际结果、网络监测、模型只读挂载、真实课件／业务验收。
