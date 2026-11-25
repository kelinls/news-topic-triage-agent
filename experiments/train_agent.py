#!/usr/bin/env python3
"""Train and evaluate the reproducible news triage agent."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from news_agent.artifacts import ROOT
from news_agent.classifier import AgentPolicy, NewsTriageAgent, neighbor_distribution, softmax


RESULTS = ROOT / "agent_results"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET_SELECTIVE_ACCURACY = 0.90
CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.guns",
]
CATEGORY_LABELS_ZH = {
    "alt.atheism": "无神论",
    "comp.graphics": "计算机图形",
    "rec.autos": "汽车",
    "rec.sport.baseball": "棒球",
    "sci.med": "医学",
    "talk.politics.guns": "枪支政治",
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def fetch_data(subset: str):
    return fetch_20newsgroups(
        subset=subset,
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
        download_if_missing=True,
    )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=30_000,
        sublinear_tf=True,
    )


def fit_model_with_grid_search(train_documents: list[str], train_labels: np.ndarray):
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("svm", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"svm__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=4,
        return_train_score=False,
    )
    search.fit(train_documents, train_labels)
    best_pipeline = search.best_estimator_
    return (
        best_pipeline.named_steps["tfidf"],
        best_pipeline.named_steps["svm"],
        search,
    )


def evaluate_probabilities(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    predictions = np.argmax(probs, axis=1)
    return (
        float(accuracy_score(labels, predictions)),
        float(log_loss(labels, probs, labels=np.arange(probs.shape[1]))),
    )


def select_probability_policy(
    model_scores: np.ndarray,
    neighbor_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best = None
    for temperature in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        model_probs = softmax(model_scores, temperature)
        for neighbor_weight in (0.0, 0.15, 0.30):
            blended = (1.0 - neighbor_weight) * model_probs + neighbor_weight * neighbor_probs
            accuracy, loss = evaluate_probabilities(blended, labels)
            rows.append(
                {
                    "temperature": temperature,
                    "neighbor_weight": neighbor_weight,
                    "accuracy": accuracy,
                    "log_loss": loss,
                }
            )
            candidate = (loss, -accuracy, temperature, neighbor_weight)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    frame = pd.DataFrame(rows).sort_values(["log_loss", "accuracy"], ascending=[True, False])
    return float(best[2]), float(best[3]), frame


def find_routing_policy(
    model_scores: np.ndarray,
    neighbor_probs: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    neighbor_weight: float,
    require_consensus: bool,
    target_accuracy: float = TARGET_SELECTIVE_ACCURACY,
) -> tuple[AgentPolicy, pd.DataFrame]:
    model_probs = softmax(model_scores, temperature)
    blended = (1.0 - neighbor_weight) * model_probs + neighbor_weight * neighbor_probs
    predictions = np.argmax(blended, axis=1)
    confidence = np.max(blended, axis=1)
    neighbor_labels = np.argmax(neighbor_probs, axis=1)
    agreement = np.max(neighbor_probs, axis=1)

    rows = []
    for confidence_threshold in np.arange(0.20, 0.96, 0.01):
        agreements = (0.0, 0.45, 0.55, 0.65, 0.75, 0.85) if require_consensus else (0.0,)
        for agreement_threshold in agreements:
            accepted = confidence >= confidence_threshold
            if require_consensus:
                accepted &= predictions == neighbor_labels
                accepted &= agreement >= agreement_threshold
            coverage = float(np.mean(accepted))
            selective_accuracy = (
                float(np.mean(predictions[accepted] == labels[accepted]))
                if np.any(accepted)
                else 0.0
            )
            rows.append(
                {
                    "confidence_threshold": round(float(confidence_threshold), 2),
                    "agreement_threshold": float(agreement_threshold),
                    "coverage": coverage,
                    "selective_accuracy": selective_accuracy,
                    "accepted": int(np.sum(accepted)),
                }
            )

    frame = pd.DataFrame(rows)
    eligible = frame[frame["selective_accuracy"] >= target_accuracy]
    if eligible.empty:
        eligible = frame.sort_values(
            ["selective_accuracy", "coverage"], ascending=[False, False]
        ).head(20)
    selected = eligible.sort_values(
        ["coverage", "selective_accuracy"], ascending=[False, False]
    ).iloc[0]
    policy = AgentPolicy(
        temperature=temperature,
        neighbor_weight=neighbor_weight,
        confidence_threshold=float(selected["confidence_threshold"]),
        agreement_threshold=float(selected["agreement_threshold"]),
        require_consensus=require_consensus,
        top_k=5,
    )
    return policy, frame


def calculate_neighbor_probs(
    query_matrix,
    train_matrix,
    train_labels: np.ndarray,
    num_classes: int,
    top_k: int = 5,
) -> np.ndarray:
    rows = []
    for row in query_matrix:
        similarities = (train_matrix @ row.T).toarray().reshape(-1)
        similarities = np.clip(similarities, 0.0, None)
        distribution, _ = neighbor_distribution(
            similarities, train_labels, num_classes, top_k
        )
        rows.append(distribution)
    return np.vstack(rows)


def policy_metrics(
    predictions: np.ndarray,
    accepted: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    accepted = np.asarray(accepted, dtype=bool)
    errors = predictions != labels
    auto_errors = int(np.sum(accepted & errors))
    accepted_count = int(np.sum(accepted))
    total_errors = int(np.sum(errors))
    return {
        "coverage": float(np.mean(accepted)),
        "review_rate": float(1.0 - np.mean(accepted)),
        "accepted_count": accepted_count,
        "review_count": int(len(labels) - accepted_count),
        "accepted_accuracy": (
            float(np.mean(predictions[accepted] == labels[accepted]))
            if accepted_count
            else 0.0
        ),
        "auto_errors": auto_errors,
        "error_capture_rate": (
            float(np.sum(accepted & errors) == 0)
            if total_errors == 0
            else float(np.sum(~accepted & errors) / total_errors)
        ),
    }


def accepted_mask_for_agent(
    model_scores: np.ndarray,
    neighbor_probs: np.ndarray,
    policy: AgentPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_probs = softmax(model_scores, policy.temperature)
    blended = (
        (1.0 - policy.neighbor_weight) * model_probs
        + policy.neighbor_weight * neighbor_probs
    )
    predictions = np.argmax(blended, axis=1)
    confidence = np.max(blended, axis=1)
    neighbor_labels = np.argmax(neighbor_probs, axis=1)
    agreement = np.max(neighbor_probs, axis=1)
    accepted = confidence >= policy.confidence_threshold
    if policy.require_consensus:
        accepted &= predictions == neighbor_labels
        accepted &= agreement >= policy.agreement_threshold
    return predictions, confidence, agreement, accepted


def choose_model_only_policy(frame: pd.DataFrame, temperature: float) -> AgentPolicy:
    eligible = frame[frame["selective_accuracy"] >= TARGET_SELECTIVE_ACCURACY]
    if eligible.empty:
        selected = frame.sort_values(
            ["selective_accuracy", "coverage"], ascending=[False, False]
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["coverage", "selective_accuracy"], ascending=[False, False]
        ).iloc[0]
    return AgentPolicy(
        temperature=temperature,
        neighbor_weight=0.0,
        confidence_threshold=float(selected["confidence_threshold"]),
        agreement_threshold=0.0,
        require_consensus=False,
        top_k=5,
    )


def plot_policy_tradeoff(
    model_frame: pd.DataFrame,
    agent_frame: pd.DataFrame,
    baseline_accuracy: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for frame, label, color in (
        (model_frame, "SVM confidence only", "#356C8A"),
        (agent_frame, "Agent: SVM + evidence gate", "#C86A45"),
    ):
        curve = (
            frame[frame["confidence_threshold"] == frame["confidence_threshold"].min()]
            if False
            else frame
        )
        grouped = (
            curve.groupby("coverage", as_index=False)["selective_accuracy"].max()
            .sort_values("coverage")
        )
        ax.plot(
            grouped["coverage"],
            grouped["selective_accuracy"],
            marker="o",
            markersize=3.4,
            linewidth=2,
            label=label,
            color=color,
        )
    ax.axhline(
        baseline_accuracy,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="Always-route SVM accuracy",
    )
    ax.axhline(
        TARGET_SELECTIVE_ACCURACY,
        color="#7B8B3D",
        linestyle=":",
        linewidth=1.5,
        label="Reliability target (0.90)",
    )
    ax.set_xlabel("Automatically routed coverage")
    ax.set_ylabel("Accuracy on auto-routed articles")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.70, 1.01)
    ax.set_title("Selective-routing trade-off")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "policy_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_outcomes(agent_metrics: dict, model_metrics: dict) -> None:
    labels = ["Accuracy", "Coverage", "Error capture"]
    model_values = [
        model_metrics["accepted_accuracy"],
        model_metrics["coverage"],
        model_metrics["error_capture_rate"],
    ]
    agent_values = [
        agent_metrics["accepted_accuracy"],
        agent_metrics["coverage"],
        agent_metrics["error_capture_rate"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars1 = ax.bar(x - width / 2, model_values, width, label="SVM threshold", color="#7FA7BD")
    bars2 = ax.bar(x + width / 2, agent_values, width, label="Full agent", color="#D08A68")
    ax.bar_label(bars1, fmt="%.3f", fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Agent routing outcomes on the hold-out test set")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "routing_outcomes.png", dpi=220)
    plt.close(fig)


def plot_dataset_distribution(train_labels: np.ndarray, test_labels: np.ndarray, names: list[str]) -> None:
    train_counts = np.bincount(train_labels, minlength=len(names))
    test_counts = np.bincount(test_labels, minlength=len(names))
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bars1 = ax.bar(x - width / 2, train_counts, width, label="Training pool", color="#356C8A")
    bars2 = ax.bar(x + width / 2, test_counts, width, label="Test", color="#D08A68")
    ax.bar_label(bars1, fontsize=8)
    ax.bar_label(bars2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Documents")
    ax.set_title("Documents by topic")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "dataset_distribution.png", dpi=220)
    plt.close(fig)


def plot_confusion(labels: np.ndarray, predictions: np.ndarray, names: list[str]) -> None:
    matrix = confusion_matrix(labels, predictions)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        RESULTS / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    display = ConfusionMatrixDisplay(matrix, display_labels=names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=28)
    ax.set_title("SVM predictions on the hold-out test set")
    ax.set_xlabel("Predicted topic")
    ax.set_ylabel("True topic")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=220)
    plt.close(fig)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_demo_cases(
    documents: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    statuses: list[str],
    target_names: list[str],
    confidences: np.ndarray,
) -> list[dict]:
    cases = []
    auto_candidates = [
        index
        for index, status in enumerate(statuses)
        if status == "auto_route"
        and predictions[index] == labels[index]
        and 200 <= len(documents[index].strip()) <= 1500
    ]
    review_candidates = [
        index
        for index, status in enumerate(statuses)
        if status == "human_review"
        and predictions[index] != labels[index]
        and 200 <= len(documents[index].strip()) <= 1500
    ]
    picks = []
    if auto_candidates:
        picks.append(("自动分流示例", max(auto_candidates, key=lambda index: confidences[index])))
    if review_candidates:
        picks.append(("人工复核示例", max(review_candidates, key=lambda index: confidences[index])))
    for name, index in picks:
        cases.append(
            {
                "name": name,
                "expected_label": target_names[int(labels[index])],
                "text": documents[index],
            }
        )
    cases.append(
        {
            "name": "异常输入示例",
            "expected_label": None,
            "text": "Too short.",
        }
    )
    return cases


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    set_seed()
    sns.set_theme(style="whitegrid", context="notebook")

    print("Loading 20 Newsgroups...")
    raw_train = fetch_data("train")
    test = fetch_data("test")
    target_names = list(raw_train.target_names)
    train_valid = np.asarray([len(text.strip()) >= 40 for text in raw_train.data])
    train_documents = [
        text
        for text, valid in zip(raw_train.data, train_valid, strict=True)
        if valid
    ]
    labels = np.asarray(raw_train.target)[train_valid]
    test_valid = np.asarray([len(text.strip()) >= 40 for text in test.data])
    print(
        f"Raw training pool: {len(raw_train.data)}; retained: {len(train_documents)}; "
        f"hold-out test: {len(test.data)}"
    )

    fit_documents, validation_documents, y_fit, y_validation = train_test_split(
        train_documents,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Development fit: {len(fit_documents)}; validation: {len(validation_documents)}")

    print("Grid-searching the SVM on the development split...")
    vectorizer, classifier, search = fit_model_with_grid_search(
        list(fit_documents), np.asarray(y_fit)
    )
    validation_matrix = vectorizer.transform(validation_documents)
    validation_scores = classifier.decision_function(validation_matrix)
    validation_neighbors = calculate_neighbor_probs(
        validation_matrix,
        vectorizer.transform(fit_documents),
        np.asarray(y_fit),
        len(target_names),
    )

    temperature, neighbor_weight, probability_frame = select_probability_policy(
        validation_scores,
        validation_neighbors,
        np.asarray(y_validation),
    )
    print(
        f"Selected temperature={temperature:g}, neighbor_weight={neighbor_weight:g}"
    )
    agent_policy, agent_policy_frame = find_routing_policy(
        validation_scores,
        validation_neighbors,
        np.asarray(y_validation),
        temperature,
        neighbor_weight,
        require_consensus=True,
    )
    model_policy, model_policy_frame = find_routing_policy(
        validation_scores,
        validation_neighbors,
        np.asarray(y_validation),
        temperature,
        0.0,
        require_consensus=False,
    )
    print(
        "Agent policy: "
        f"confidence>={agent_policy.confidence_threshold:.2f}, "
        f"agreement>={agent_policy.agreement_threshold:.2f}"
    )

    print("Refitting the deployment model on all training documents...")
    final_vectorizer = build_vectorizer()
    final_train_matrix = final_vectorizer.fit_transform(train_documents)
    final_classifier = LinearSVC(C=float(search.best_params_["svm__C"]), random_state=RANDOM_STATE)
    final_classifier.fit(final_train_matrix, labels)

    test_matrix = final_vectorizer.transform(test.data)
    test_scores = final_classifier.decision_function(test_matrix)
    test_neighbors = calculate_neighbor_probs(
        test_matrix,
        final_train_matrix,
        labels,
        len(target_names),
    )

    model_predictions, model_confidence, _, model_accepted = accepted_mask_for_agent(
        test_scores, test_neighbors, model_policy
    )
    agent_predictions, agent_confidence, agent_agreement, agent_accepted = (
        accepted_mask_for_agent(test_scores, test_neighbors, agent_policy)
    )
    model_accepted &= test_valid
    agent_accepted &= test_valid
    baseline_predictions = np.argmax(softmax(test_scores, temperature), axis=1)
    baseline = {
        "accuracy": float(accuracy_score(test.target, baseline_predictions)),
        "macro_f1": float(
            f1_score(test.target, baseline_predictions, average="macro")
        ),
    }
    model_metrics = policy_metrics(
        model_predictions, model_accepted, np.asarray(test.target)
    )
    agent_metrics = policy_metrics(
        agent_predictions, agent_accepted, np.asarray(test.target)
    )

    plot_dataset_distribution(labels, np.asarray(test.target), target_names)
    plot_confusion(np.asarray(test.target), baseline_predictions, target_names)
    classification = pd.DataFrame(
        classification_report(
            test.target,
            baseline_predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    classification.to_csv(RESULTS / "classification_report.csv", encoding="utf-8-sig")

    class_rows = []
    for class_index, class_name in enumerate(target_names):
        class_mask = np.asarray(test.target) == class_index
        accepted_in_class = class_mask & agent_accepted
        class_rows.append(
            {
                "topic": class_name,
                "test_count": int(np.sum(class_mask)),
                "baseline_precision": float(classification.loc[class_name, "precision"]),
                "baseline_recall": float(classification.loc[class_name, "recall"]),
                "baseline_f1": float(classification.loc[class_name, "f1-score"]),
                "auto_coverage": float(np.mean(agent_accepted[class_mask])),
                "accepted_accuracy": (
                    float(np.mean(agent_predictions[accepted_in_class] == class_index))
                    if np.any(accepted_in_class)
                    else 0.0
                ),
            }
        )
    pd.DataFrame(class_rows).to_csv(
        RESULTS / "classwise_metrics.csv", index=False, encoding="utf-8-sig"
    )

    prediction_frame = pd.DataFrame(
        {
            "true_label": np.asarray(target_names)[test.target],
            "predicted_label": np.asarray(target_names)[agent_predictions],
            "routing_status": ["auto_route" if accepted else "human_review" for accepted in agent_accepted],
            "confidence": agent_confidence,
            "neighbor_agreement": agent_agreement,
            "correct": agent_predictions == np.asarray(test.target),
            "excerpt": [" ".join(text.split())[:300] for text in test.data],
        }
    )
    prediction_frame.to_csv(RESULTS / "test_predictions.csv", index=False, encoding="utf-8-sig")

    statuses = ["auto_route" if accepted else "human_review" for accepted in agent_accepted]
    demo_cases = select_demo_cases(
        test.data,
        np.asarray(test.target),
        agent_predictions,
        statuses,
        target_names,
        agent_confidence,
    )
    save_json(RESULTS / "demo_cases.json", {"cases": demo_cases})

    agent = NewsTriageAgent(
        vectorizer=final_vectorizer,
        classifier=final_classifier,
        train_matrix=final_train_matrix,
        train_documents=list(train_documents),
        train_labels=labels,
        target_names=target_names,
        label_names_zh=CATEGORY_LABELS_ZH,
        policy=agent_policy,
    )
    artifact = {
        "agent": agent,
        "model_version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "20 Newsgroups, six-category subset",
        "model": "TF-IDF + LinearSVC",
    }
    joblib.dump(artifact, ARTIFACTS / "news_triage_agent.joblib", compress=3)

    policy_frame = pd.concat(
        [
            model_policy_frame.assign(policy="svm_threshold"),
            agent_policy_frame.assign(policy="full_agent"),
        ],
        ignore_index=True,
    )
    policy_frame.to_csv(RESULTS / "policy_grid.csv", index=False, encoding="utf-8-sig")
    probability_frame.to_csv(
        RESULTS / "probability_selection.csv", index=False, encoding="utf-8-sig"
    )

    errors = pd.DataFrame(
        {
            "true_label": np.asarray(target_names)[test.target],
            "predicted_label": np.asarray(target_names)[agent_predictions],
            "routing_status": statuses,
            "confidence": agent_confidence,
            "neighbor_agreement": agent_agreement,
            "excerpt": [" ".join(text.split())[:300] for text in test.data],
        }
    )
    errors[errors["true_label"] != errors["predicted_label"]].to_csv(
        RESULTS / "errors.csv", index=False, encoding="utf-8-sig"
    )

    plot_policy_tradeoff(model_policy_frame, agent_policy_frame, baseline["accuracy"])
    plot_outcomes(agent_metrics, model_metrics)

    metrics = {
        "dataset": "20 Newsgroups, six categories",
        "categories": CATEGORIES,
        "raw_train_size": len(raw_train.data),
        "train_size": len(train_documents),
        "excluded_short_train_documents": int(np.sum(~train_valid)),
        "development_fit_size": len(fit_documents),
        "validation_size": len(validation_documents),
        "test_size": len(test.data),
        "tfidf_features": len(final_vectorizer.vocabulary_),
        "best_C": float(search.best_params_["svm__C"]),
        "development_cv_macro_f1": float(search.best_score_),
        "temperature": temperature,
        "neighbor_weight": neighbor_weight,
        "target_selective_accuracy": TARGET_SELECTIVE_ACCURACY,
        "agent_policy": agent_policy.to_dict(),
        "model_only_policy": model_policy.to_dict(),
        "baseline": baseline,
        "model_only": model_metrics,
        "full_agent": agent_metrics,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    save_json(RESULTS / "agent_metrics.json", metrics)

    lines = [
        "News triage agent — reproducible run",
        "-----------------------------------",
        f"Training pool: {len(train_documents)} retained from {len(raw_train.data)}",
        f"Validation: {len(validation_documents)}; test: {len(test.data)}",
        f"TF-IDF features: {metrics['tfidf_features']}",
        f"SVM C={metrics['best_C']:g}; development CV macro-F1={metrics['development_cv_macro_f1']:.4f}",
        f"Baseline accuracy: {baseline['accuracy']:.4f}; macro-F1: {baseline['macro_f1']:.4f}",
        f"Agent policy: confidence>={agent_policy.confidence_threshold:.2f}, agreement>={agent_policy.agreement_threshold:.2f}",
        f"SVM threshold: coverage={model_metrics['coverage']:.4f}, selective accuracy={model_metrics['accepted_accuracy']:.4f}",
        f"Full agent: coverage={agent_metrics['coverage']:.4f}, selective accuracy={agent_metrics['accepted_accuracy']:.4f}",
        f"Error capture: {agent_metrics['error_capture_rate']:.4f}",
        f"Elapsed: {metrics['runtime_seconds']:.1f} seconds",
    ]
    (RESULTS / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
