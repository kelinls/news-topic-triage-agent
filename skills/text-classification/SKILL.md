---
name: news-topic-classification
description: "Guide an LLM controller through six-way news topic triage with an SVM tool, retrieval evidence, uncertainty review, and audit logging."
argument-hint: "[article text to triage]"
source: "https://github.com/brycewang-stanford/Auto-Empirical-Research-Skills/blob/7c85ad2c0515b83148b1084be373effbc265ae3b/skills/54-scdenney-open-science-skills/skills/text-classification/SKILL.md"
license: "CC BY-SA 4.0"
upstream_sha256: "58237778185b9ac368600f5d33d31f3118c94bc8bb063c44d070d04770d25df7"
---

# News Topic Classification Skill

This is a task-specific adaptation of the upstream skill. The upstream skill
describes how to operationalize an LLM text-classification pipeline with a
codebook, validation, uncertainty handling, and reporting. This version fixes
the six labels and treats a trained SVM and neighbor retrieval as observable
tools rather than as ground truth.

## Codebook

Use exactly one of these labels:

- `alt.atheism`: beliefs, arguments, institutions, or communities concerning
  atheism and religion. Include theological debate and church-state discussion;
  exclude general political firearms policy unless religion is central.
- `comp.graphics`: computer image generation, rendering, formats, graphics
  files, visualization software, and hardware used for graphics. Include
  technical image-processing discussion; exclude photography with no computing
  angle.
- `rec.autos`: cars, engines, maintenance, buying, driving, motorsport, and
  automotive products. Include product/repair discussion; exclude baseball
  games or political regulation unless automobiles are central.
- `rec.sport.baseball`: baseball teams, players, games, statistics, and league
  discussion. Include historical and fantasy discussion; exclude other sports.
- `sci.med`: medicine, disease, treatment, anatomy, health research, and
  clinical practice. Include patient/physician perspectives; exclude computer
  graphics used for medical imaging when the computing topic dominates.
- `talk.politics.guns`: firearms policy, gun control, legislation, rights, and
  political conflict concerning firearms. Include constitutional or public
  safety arguments; exclude crime stories where policy is not the subject.

If the text is empty, shorter than 40 characters, over 50,000 characters, or
otherwise uncodeable, use the terminal outcome `rejected` rather than forcing a
topic.

## Tool Use

1. Load this skill before using the task-specific tools.
2. Validate the article before classification. Do not echo secrets or infer a
   label from an invalid input.
3. Call `classify_topic` to obtain the SVM candidate, margin, confidence, and
   positively contributing features.
4. Call `retrieve_evidence` and compare the candidate with the retrieved label
   distribution. A retrieval result is evidence, not truth.
5. Use `auto_route` only when the model and evidence agree and the combined
   support is sufficient. Send borderline, conflicting, or short-but-valid
   cases to `request_human_review` with a concrete reason.
6. Never invent a tool result. If a tool fails, retry once when appropriate;
   otherwise degrade to human review or rejection with the failure recorded.
7. Stop only after `submit_result` or `request_human_review`.

## Quality Checks

- The returned label must be in the codebook exactly.
- Rationale must cite observed model/evidence signals, not hidden ground truth.
- Low confidence or model/evidence disagreement must not be silently accepted.
- Every run must remain within the configured step limit and produce a trace.
- Evaluation must report completion, routing quality, human-review rate,
  latency, tool failures, and representative errors.

