#!/usr/bin/env python3
"""Focused tests for the news triage agent."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from news_agent.classifier import AgentPolicy, NewsTriageAgent, softmax


TRAIN_DOCS = [
    "graphics rendering polygon texture shader computer image",
    "graphics pixel raster animation rendering image",
    "baseball pitcher inning home run bat stadium",
    "baseball league player hit game score",
]
TRAIN_LABELS = np.asarray([0, 0, 1, 1])
TARGET_NAMES = ["comp.graphics", "rec.sport.baseball"]
LABELS_ZH = {"comp.graphics": "计算机图形", "rec.sport.baseball": "棒球"}


def make_agent(policy: AgentPolicy) -> NewsTriageAgent:
    vectorizer = TfidfVectorizer()
    train_matrix = vectorizer.fit_transform(TRAIN_DOCS)
    classifier = LinearSVC(random_state=42)
    classifier.fit(train_matrix, TRAIN_LABELS)
    return NewsTriageAgent(
        vectorizer=vectorizer,
        classifier=classifier,
        train_matrix=train_matrix,
        train_documents=TRAIN_DOCS,
        train_labels=TRAIN_LABELS,
        target_names=TARGET_NAMES,
        label_names_zh=LABELS_ZH,
        policy=policy,
        min_input_chars=10,
    )


def test_softmax_rows_are_probabilities() -> None:
    probabilities = softmax(np.asarray([[2.0, 1.0], [-1.0, 3.0]]), 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)


def test_short_input_is_rejected() -> None:
    agent = make_agent(AgentPolicy())
    result = agent.triage("bad")
    assert result["status"] == "rejected"
    assert result["suggested_label"] is None


def test_clear_article_is_routed_with_evidence() -> None:
    agent = make_agent(
        AgentPolicy(
            temperature=1.0,
            neighbor_weight=0.25,
            confidence_threshold=0.30,
            agreement_threshold=0.45,
            require_consensus=True,
        )
    )
    result = agent.triage(
        "Computer graphics rendering uses polygon geometry and a texture shader."
    )
    assert result["status"] == "auto_route"
    assert result["suggested_label"] == "comp.graphics"
    assert result["neighbors"]
    assert result["top_features"]


def test_high_threshold_sends_article_to_review() -> None:
    agent = make_agent(
        AgentPolicy(confidence_threshold=1.01, require_consensus=False)
    )
    result = agent.triage("Computer graphics rendering uses a texture shader.")
    assert result["status"] == "human_review"


if __name__ == "__main__":
    test_softmax_rows_are_probabilities()
    test_short_input_is_rejected()
    test_clear_article_is_routed_with_evidence()
    test_high_threshold_sends_article_to_review()
    print("4 tests passed")
