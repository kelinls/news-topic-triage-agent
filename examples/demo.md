# AgentX API 演示

服务启动后运行 `python -m ml.demo`。

这次实测向 `POST /api/triage` 提交自写的图形学文本。

> The graphics team benchmarked a GPU renderer on a scene with textured polygons. The scene uses vertex shaders for geometry and a raster pipeline for the final image.

| 顺序 | 工具 | 观察 |
| --- | --- | --- |
| 1 | `Skill` | name=news-topic-classification |
| 2 | `classify_topic` | label=comp.graphics, confidence=0.998359, margin=0.996923 |
| 3 | `retrieve_evidence` | label=comp.graphics, agreement=1.0 |
| 4 | `submit_route` | status=auto_route, label=comp.graphics |

API 返回 `auto_route`，标签 `comp.graphics`，用时 6241 毫秒。

完整实测响应见 [demo-response.json](demo-response.json)。

[实跑截图](demo-screenshot.png)记录了使用同一篇文章的另一次请求，可查看输入、四次工具调用、分流结果和理由。
