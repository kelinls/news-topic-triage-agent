# 新闻主题智能分诊 Agent

基于 Spring Boot 与 Spring AI AgentX 的新闻分诊后端。提交英文文章后，Agent 加载分类 Skill，调用本地 TF-IDF 与 LinearSVC、近邻检索和分流工具，返回分类建议、人工复核或拒绝，以及可检查的工具轨迹。

## 源码结构

| 模块 | 职责 |
| --- | --- |
| [backend](backend/) | Java 21 / Spring Boot API，请求状态、模型配置、业务工具和针对性测试 |
| [vendor/spring-ai-agentx](vendor/spring-ai-agentx/) | 原有 AgentX Core 的源代码依赖，包含 ReAct 执行引擎、DeepSeek 模型适配和 SkillsTool |
| [ml](ml/) | Python 训练、回环推理服务和成对实验脚本 |
| [skills](skills/) | 新闻主题判例、上游来源与许可证 |
| [examples](examples/) | 独立演示文本和实际 API 执行记录 |

请求由 `TriageController` 转交 `TriageService`。每次请求构建独立的 `ReactAgent` 和工具状态，AgentX 执行模型与工具的循环。分类和检索工具通过回环 HTTP 请求访问 ML 服务。自动分流要求主题在两种证据中一致，并达到服务端阈值。模型建议复核或 Agent 流程异常时，API 返回 `human_review`。原型把结果交给调用方，人工复核状态表示需要编辑处理。

## 本地运行

需要 JDK 21、Maven 3.9 和 Python 3.12。首次训练会下载 20 Newsgroups 六类语料。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements.txt
python -m ml.train
python -m ml.service
```

在另一终端设置 `DEEPSEEK_API_KEY` 环境变量，再构建并启动 API。可选配置见 [`.env.example`](.env.example)。两个服务默认只监听回环地址。

```bash
mvn -B -ntp clean package
java -jar backend/target/news-triage-backend-1.0.0.jar
```

```bash
curl -fsS http://127.0.0.1:8088/api/health
python -m ml.demo
```

API 接收 40 至 50,000 字符的英文正文。文章会发送给配置的模型供应商，检索工具只返回主题分布。请求和响应示例见 [演示记录](examples/demo.md)。

## 实验复现

```bash
mvn -B -ntp test
python -m ml.evaluate --workers 3
```

评估使用相同的 24 篇样本对比固定基线、AgentX 和关闭 Skill 的 AgentX，另选 12 篇做重复检查。训练指标和逐例实验记录生成于 `ml/results/`。[汇总实验数据](examples/evaluation.json)保存报告中的指标。官方测试集的 LinearSVC 准确率为 83.99%，宏 F1 为 83.68%。端到端样本用于观察分流和异常处理，样本量及构造方式适合复现实验对照。

## 来源和许可

- AgentX Core 源码快照和 Apache-2.0 许可证见 [上游声明](vendor/spring-ai-agentx/UPSTREAM.md)。
- 文本分类 Skill 改编信息和 CC BY-SA 4.0 许可证见 [NOTICE](skills/text-classification/NOTICE.md)。
- scikit-learn 提供 20 Newsgroups 下载接口。含语料的模型文件和逐例运行记录由本地脚本生成。
