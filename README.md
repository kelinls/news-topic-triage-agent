# 新闻主题智能分诊 Agent

这个命令行原型将英文新闻、论坛帖和工单分到六个主题。DeepSeek 控制器加载文本分类 Skill，按需调用输入校验、TF-IDF 与 LinearSVC 分类、近邻检索和终止工具。输出状态为 `auto_route`（自动分流）、`human_review`（人工复核）或 `rejected`（拒绝输入）。

## 安装与训练

在仓库根目录使用 Python 3.12 执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_agent.py
```

训练脚本通过 scikit-learn 获取 20 Newsgroups，并生成模型和检索索引。首次训练需要下载数据。输入接受 40 至 50,000 字符的英文文本。

## 运行演示

先在当前终端设置环境变量 `DEEPSEEK_API_KEY`。可选的模型和接口配置见 [`.env.example`](.env.example)。

```bash
python news_llm_agent.py --file examples/graphics.txt --trace
```

输出包括候选主题、理由、已加载的 Skill 和工具调用记录。[真实运行记录](examples/demo.md)展示输入、关键观察及最终结果。演示需要联网调用 DeepSeek，输入以及检索工具的观察结果会发送给所配置的模型服务。

固定门控基线使用相同的本地模型，运行时无需 API：

```bash
python news_triage_agent.py --file examples/graphics.txt
```

## 测试与评估

完成训练后执行以下离线测试：

```bash
python test_agent.py
python test_llm_agent.py
python test_experiment.py
```

设置 API 密钥后，运行 `python evaluate_llm_agent.py --run all` 复现完整 Agent、关闭 Skill 的消融和重复一致性检查。脚本根据固定种子生成 24 个测试任务，结果写入 `llm_agent_results/`。基线指标由训练脚本写入 `agent_results/agent_metrics.json`。

LinearSVC 在 2,261 篇官方测试文本上的准确率为 83.99%，宏 F1 为 83.68%。在另一次 24 个端到端任务的评估中，完整 Agent 正确自动分流 13 个任务，转人工 8 个，拒绝 3 个。该评估包含边界和挑战样本，用于观察路由行为。端到端结果会受模型版本和服务状态影响。

## 组件与来源

- **机器学习**使用 scikit-learn 的 TF-IDF、LinearSVC 和近邻检索。数据为 20 Newsgroups 六类官方训练集与测试集。原始帖子、包含语料的模型和逐例实验轨迹在本地生成。
- **控制器**使用 DeepSeek 兼容 Chat Completions 接口，默认模型为 `deepseek-flash`。
- **文本分类 Skill**改编自 Auto-Empirical-Research-Skills，版本、来源和改动见 [复用说明](skills/text-classification/NOTICE.md)，采用 [CC BY-SA 4.0](skills/text-classification/LICENSE)。
