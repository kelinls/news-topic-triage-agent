# 图形主题分流演示

这次运行使用自写英文文本和 DeepSeek `deepseek-flash`。以下内容摘自实际 CLI 输出。

```bash
python news_llm_agent.py --file examples/graphics.txt --trace
```

输入文本

> The graphics team benchmarked a GPU renderer on a scene with textured polygons. The scene uses vertex shaders for geometry and a raster pipeline for the final image.

| LLM 步骤 | 工具 | 观察结果 |
| --- | --- | --- |
| 1 | `list_skills` | 发现 `news-topic-classification` |
| 2 | `load_skill` | 成功加载分类 Skill |
| 3 | `validate_text` | 通过长度校验 |
| 4 | `classify_topic` | 候选 `comp.graphics`，SVM 展示分数 0.998 |
| 4 | `retrieve_evidence` | 近邻多数主题为 `comp.graphics`，一致率 100% |
| 5 | `submit_result` | 自动分流到计算机图形主题 |

```json
{
  "status": "auto_route",
  "suggested_label": "comp.graphics",
  "steps": 5,
  "tool_calls": 6,
  "fallback": false
}
```
