#!/usr/bin/env python3
"""Core runtime for the news triage agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass
class AgentPolicy:
    temperature: float = 1.0
    neighbor_weight: float = 0.0
    confidence_threshold: float = 0.60
    agreement_threshold: float = 0.60
    require_consensus: bool = True
    top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=float) / max(float(temperature), 1e-9)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cosine_similarities(
    query_matrix: sparse.spmatrix,
    train_matrix: sparse.spmatrix,
) -> np.ndarray:
    scores = (train_matrix @ query_matrix.T).toarray().reshape(-1)
    return np.clip(scores, 0.0, None)


def neighbor_distribution(
    similarities: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_k = min(int(top_k), len(similarities))
    order = np.argsort(similarities)[::-1][:top_k]
    weights = similarities[order].astype(float)
    if np.sum(weights) <= 1e-12:
        weights = np.ones_like(weights)
    distribution = np.zeros(num_classes, dtype=float)
    for index, weight in zip(order, weights, strict=True):
        distribution[int(train_labels[index])] += float(weight)
    distribution /= np.sum(distribution)
    return distribution, order


def _compact_excerpt(text: str, limit: int = 220) -> str:
    return " ".join(text.split())[:limit]


class NewsTriageAgent:
    """Classify an article, retrieve supporting examples, and gate routing."""

    def __init__(
        self,
        vectorizer,
        classifier,
        train_matrix: sparse.spmatrix,
        train_documents: list[str],
        train_labels: np.ndarray,
        target_names: list[str],
        label_names_zh: dict[str, str],
        policy: AgentPolicy,
        min_input_chars: int = 40,
        max_input_chars: int = 50_000,
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.train_matrix = train_matrix
        self.train_documents = train_documents
        self.train_labels = np.asarray(train_labels)
        self.target_names = list(target_names)
        self.label_names_zh = dict(label_names_zh)
        self.policy = policy
        self.min_input_chars = int(min_input_chars)
        self.max_input_chars = int(max_input_chars)

    def _rejected(self, message: str) -> dict[str, Any]:
        return {
            "status": "rejected",
            "suggested_label": None,
            "suggested_label_zh": None,
            "confidence": 0.0,
            "neighbor_agreement": 0.0,
            "reason": message,
            "top_features": [],
            "neighbors": [],
            "steps": [
                {"step": "validate_input", "result": "rejected", "detail": message}
            ],
        }

    def triage(self, text: str) -> dict[str, Any]:
        text = "" if text is None else str(text)
        text = text.replace("\x00", " ").strip()
        if len(text) < self.min_input_chars:
            return self._rejected(f"文本长度不足 {self.min_input_chars} 个字符")
        if len(text) > self.max_input_chars:
            return self._rejected(f"文本超过 {self.max_input_chars} 个字符上限")

        query_matrix = self.vectorizer.transform([text])
        model_scores = np.asarray(self.classifier.decision_function(query_matrix))
        if model_scores.ndim == 1:
            model_scores = model_scores.reshape(1, -1)
        model_probs = softmax(model_scores, self.policy.temperature)[0]

        similarities = cosine_similarities(query_matrix, self.train_matrix)
        neighbor_probs, neighbor_order = neighbor_distribution(
            similarities,
            self.train_labels,
            len(self.target_names),
            self.policy.top_k,
        )
        blended_probs = (
            (1.0 - self.policy.neighbor_weight) * model_probs
            + self.policy.neighbor_weight * neighbor_probs
        )

        ordered = np.argsort(blended_probs)[::-1]
        predicted_index = int(ordered[0])
        neighbor_index = int(np.argmax(neighbor_probs))
        confidence = float(blended_probs[predicted_index])
        agreement = float(neighbor_probs[neighbor_index])
        consensus = predicted_index == neighbor_index
        accepted = confidence >= self.policy.confidence_threshold
        if self.policy.require_consensus:
            accepted = accepted and consensus and agreement >= self.policy.agreement_threshold

        status = "auto_route" if accepted else "human_review"
        if not accepted:
            reason = (
                "模型置信度或近邻一致性不足，转入人工复核"
                if confidence < self.policy.confidence_threshold
                else "模型与近邻证据不一致，转入人工复核"
            )
        else:
            reason = "置信度与近邻证据均达到自动分流阈值"

        neighbors = []
        for rank, index in enumerate(neighbor_order, start=1):
            label = self.target_names[int(self.train_labels[index])]
            neighbors.append(
                {
                    "rank": rank,
                    "label": label,
                    "label_zh": self.label_names_zh.get(label, label),
                    "similarity": round(float(similarities[index]), 6),
                    "excerpt": _compact_excerpt(self.train_documents[index]),
                }
            )

        alternatives = [
            {
                "label": self.target_names[int(index)],
                "label_zh": self.label_names_zh.get(self.target_names[int(index)], ""),
                "score": round(float(blended_probs[index]), 6),
            }
            for index in ordered[:3]
        ]
        top_features = self._top_features(query_matrix, predicted_index)
        steps = [
            {"step": "validate_input", "result": "passed", "detail": f"{len(text)} 个字符"},
            {
                "step": "classify",
                "result": self.target_names[predicted_index],
                "detail": f"confidence={confidence:.4f}",
            },
            {
                "step": "retrieve_evidence",
                "result": self.target_names[neighbor_index],
                "detail": f"agreement={agreement:.4f}, top_k={self.policy.top_k}",
            },
            {
                "step": "route",
                "result": status,
                "detail": reason,
            },
        ]

        return {
            "status": status,
            "suggested_label": self.target_names[predicted_index],
            "suggested_label_zh": self.label_names_zh.get(
                self.target_names[predicted_index], ""
            ),
            "confidence": round(confidence, 6),
            "neighbor_label": self.target_names[neighbor_index],
            "neighbor_label_zh": self.label_names_zh.get(
                self.target_names[neighbor_index], ""
            ),
            "neighbor_agreement": round(agreement, 6),
            "reason": reason,
            "alternatives": alternatives,
            "top_features": top_features,
            "neighbors": neighbors,
            "steps": steps,
        }

    def _top_features(
        self,
        query_matrix: sparse.spmatrix,
        predicted_index: int,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        coefficient_matrix = np.asarray(self.classifier.coef_)
        if coefficient_matrix.shape[0] == 1:
            coefficients = (
                -coefficient_matrix[0]
                if predicted_index == 0
                else coefficient_matrix[0]
            )
        else:
            coefficients = coefficient_matrix[predicted_index]
        coefficients = np.asarray(coefficients).ravel()
        contributions = query_matrix.multiply(coefficients).tocoo()
        if contributions.nnz == 0:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        rows = []
        for column, value in zip(contributions.col, contributions.data, strict=True):
            if value <= 0:
                continue
            rows.append(
                {
                    "feature": str(feature_names[column]),
                    "contribution": round(float(value), 6),
                }
            )
        rows.sort(key=lambda item: item["contribution"], reverse=True)
        return rows[:limit]
