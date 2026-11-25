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

## Tool Use

The API rejects texts outside the input length bounds before the Agent starts.
For accepted input, apply the codebook as follows:

1. Call `classify_topic` for the candidate label, margin and contributing terms.
2. Call `retrieve_evidence` to compare that candidate with the distribution
   of nearby training labels. Increase top-k within the supported range
   when the initial retrieved evidence is sparse.
3. Use `submit_route` only if both observations and the article's actual topic
   agree. A news release mentioning an agency, a lobby or a public official
   can look political even when its topic concerns medicine, religion or cars.
4. Use `request_review` when a label boundary is unclear, an article is outside
   the six topics, the evidence conflicts, or a needed tool remains unavailable.
5. Stop issuing tools when a terminal tool succeeds.

Support each reason with observed evidence. Treat softmax scores as ranking
signals and retrieved labels as supporting observations.
