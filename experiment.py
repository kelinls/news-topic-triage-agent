#!/usr/bin/env python3
"""News topic classification with linear and RBF-kernel SVMs.

The script downloads a six-category subset of 20 Newsgroups, compares a
TfidfVectorizer + linear SVM with an RBF-kernel SVM, performs stratified
five-fold grid search, and generates the tables and figures used in the report.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RANDOM_STATE = 42
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
VISUAL_CATEGORIES = ["comp.graphics", "rec.sport.baseball"]


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_data(categories: list[str], subset: str):
    return fetch_20newsgroups(
        subset=subset,
        categories=categories,
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


def build_search(classifier, parameters: dict, project: bool = False) -> GridSearchCV:
    steps = [("tfidf", build_vectorizer())]
    if project:
        steps.append(("pca", PCA(n_components=2, random_state=RANDOM_STATE)))
    steps.append(("clf", classifier))
    # Fit vocabulary, IDF and optional PCA only on the training part of each fold.
    return GridSearchCV(
        Pipeline(steps),
        {f"clf__{key}": value for key, value in parameters.items()},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=4,
        return_train_score=True,
    )


def metrics_row(name: str, y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]
    return {
        "模型": name,
        "准确率": accuracy,
        "宏平均精确率": macro_p,
        "宏平均召回率": macro_r,
        "宏平均F1": macro_f1,
        "加权F1": weighted_f1,
    }


def plot_dataset_distribution(train_target, test_target, target_names) -> None:
    train_counts = np.bincount(train_target, minlength=len(target_names))
    test_counts = np.bincount(test_target, minlength=len(target_names))
    x = np.arange(len(target_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, train_counts, width, label="Training", color="#356C8A")
    ax.bar(x + width / 2, test_counts, width, label="Test", color="#D79B4D")
    ax.set_xticks(x)
    ax.set_xticklabels(
        target_names,
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Documents")
    ax.set_title("Document Distribution by Newsgroup")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "dataset_distribution.png", dpi=220)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame) -> None:
    metrics = ["准确率", "宏平均精确率", "宏平均召回率", "宏平均F1"]
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, (_, row) in zip((-width / 2, width / 2), summary.iterrows()):
        values = [row[metric] for metric in metrics]
        bars = ax.bar(x + offset, values, width, label=row["模型"])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_ylim(0.75, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Test-set Performance of the Two SVM Models")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=220)
    plt.close(fig)


def plot_confusion(y_true, y_pred, target_names, filename: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        RESULTS / Path(filename).with_suffix(".csv"), encoding="utf-8-sig"
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        xticks_rotation=30,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=220)
    plt.close(fig)


def plot_rbf_grid(cv_results: pd.DataFrame) -> None:
    pivot = cv_results.pivot(
        index="param_gamma", columns="param_C", values="mean_test_score"
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.7,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("RBF SVM Five-fold Mean Macro F1")
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    fig.tight_layout()
    fig.savefig(RESULTS / "rbf_grid_heatmap.png", dpi=220)
    plt.close(fig)


def plot_linear_curve(cv_results: pd.DataFrame) -> None:
    curve = (
        cv_results.groupby("param_C", as_index=False)
        .agg(mean=("mean_test_score", "mean"), std=("std_test_score", "mean"))
        .sort_values("param_C")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(
        curve["param_C"],
        curve["mean"],
        yerr=curve["std"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#356C8A",
    )
    ax.set_xscale("log")
    ax.set_xticks(curve["param_C"])
    ax.set_xticklabels([str(value) for value in curve["param_C"]])
    ax.set_xlabel("C")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Linear SVM Hyperparameter Validation Curve")
    ax.grid(alpha=0.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "linear_curve.png", dpi=220)
    plt.close(fig)


def top_features_per_class(
    feature_names: np.ndarray, coef: np.ndarray, target_names: list[str], top_k: int = 8
) -> pd.DataFrame:
    rows = []
    for class_index, class_name in enumerate(target_names):
        order = np.argsort(coef[class_index])[::-1][:top_k]
        rows.append(
            {
                "类别": class_name,
                "关键词": ", ".join(feature_names[index] for index in order),
            }
        )
    return pd.DataFrame(rows)


def error_analysis(
    docs,
    y_true,
    y_pred,
    decision_values,
    target_names,
    max_rows: int = 10,
) -> pd.DataFrame:
    rows = []
    true_names = np.asarray(target_names)[y_true]
    pred_names = np.asarray(target_names)[y_pred]
    for index in np.flatnonzero(y_true != y_pred):
        predicted_score = float(decision_values[index, y_pred[index]])
        true_score = float(decision_values[index, y_true[index]])
        snippet = " ".join(docs[index].split())[:260]
        rows.append(
            {
                "真实类别": true_names[index],
                "预测类别": pred_names[index],
                "预测类决策值": predicted_score,
                "真实类决策值": true_score,
                "文本摘要": snippet,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("预测类决策值", ascending=False).head(max_rows)


def plot_decision_boundaries() -> dict:
    binary_train = fetch_data(VISUAL_CATEGORIES, "train")
    binary_test = fetch_data(VISUAL_CATEGORIES, "test")
    linear_search = build_search(
        LinearSVC(random_state=RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0]},
        project=True,
    )
    rbf_search = build_search(
        SVC(kernel="rbf"),
        {"C": [1.0, 10.0, 100.0], "gamma": [0.1, 1.0, 10.0]},
        project=True,
    )
    linear_search.fit(binary_train.data, binary_train.target)
    rbf_search.fit(binary_train.data, binary_train.target)
    preprocessing = linear_search.best_estimator_[:-1]
    train_2d = preprocessing.transform(binary_train.data)
    test_2d = preprocessing.transform(binary_test.data)
    vectorizer = preprocessing.named_steps["tfidf"]
    pca = preprocessing.named_steps["pca"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    scatter = None
    for ax, title, estimator in (
        (axes[0], "Linear SVM boundary", linear_search.best_estimator_.named_steps["clf"]),
        (axes[1], "RBF SVM boundary", rbf_search.best_estimator_.named_steps["clf"]),
    ):
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            train_2d,
            response_method="predict",
            alpha=0.18,
            cmap="coolwarm",
            ax=ax,
            grid_resolution=240,
        )
        scatter = ax.scatter(
            test_2d[:, 0],
            test_2d[:, 1],
            c=binary_test.target,
            cmap="coolwarm",
            s=15,
            alpha=0.72,
            edgecolors="none",
        )
        ax.set_title(title)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")

    legend_labels = list(binary_train.target_names)
    handles, _ = scatter.legend_elements()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RESULTS / "decision_boundaries.png", dpi=220)
    plt.close(fig)

    return {
        "visual_categories": VISUAL_CATEGORIES,
        "train_size": int(len(binary_train.data)),
        "test_size": int(len(binary_test.data)),
        "tfidf_features": len(vectorizer.vocabulary_),
        "linear_best_C": float(linear_search.best_params_["clf__C"]),
        "linear_cv_f1": float(linear_search.best_score_),
        "rbf_best_C": float(rbf_search.best_params_["clf__C"]),
        "rbf_best_gamma": float(rbf_search.best_params_["clf__gamma"]),
        "rbf_cv_f1": float(rbf_search.best_score_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def main() -> None:
    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    set_seed()
    sns.set_theme(style="whitegrid", context="notebook")

    print("Loading 20 Newsgroups...")
    train = fetch_data(CATEGORIES, "train")
    test = fetch_data(CATEGORIES, "test")
    target_names = list(train.target_names)
    print(f"Train documents: {len(train.data)}")
    print(f"Test documents: {len(test.data)}")
    plot_dataset_distribution(
        train.target, test.target, target_names
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "类别": name,
                "训练集数量": int(np.sum(train.target == index)),
                "测试集数量": int(np.sum(test.target == index)),
            }
            for index, name in enumerate(target_names)
        ]
    )
    dataset_summary.to_csv(
        RESULTS / "dataset_summary.csv", index=False, encoding="utf-8-sig"
    )

    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train.data)
    x_test = vectorizer.transform(test.data)
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF feature count: {x_train.shape[1]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    summaries = []
    predictions = {}

    print("Evaluating Multinomial Naive Bayes baseline...")
    nb = MultinomialNB(alpha=0.3)
    nb.fit(x_train, train.target)
    nb_pred = nb.predict(x_test)
    summaries.append(metrics_row("MultinomialNB", test.target, nb_pred))

    print("Grid-searching linear SVM...")
    linear_search = build_search(
        LinearSVC(random_state=RANDOM_STATE, max_iter=20_000),
        {"C": [0.1, 1.0, 10.0]},
    )
    linear_search.fit(train.data, train.target)
    linear_pred = linear_search.predict(test.data)
    predictions["linear"] = linear_pred
    summaries.append(metrics_row("LinearSVC", test.target, linear_pred))
    linear_cv = pd.DataFrame(linear_search.cv_results_)
    linear_cv = linear_cv.rename(columns={"param_clf__C": "param_C"})
    linear_cv.to_csv(
        RESULTS / "linear_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    plot_linear_curve(linear_cv)

    print("Grid-searching RBF SVM...")
    rbf_search = build_search(
        SVC(kernel="rbf"),
        {
            "C": [1.0, 10.0, 100.0],
            "gamma": [0.001, 0.01, 0.1],
        },
    )
    rbf_search.fit(train.data, train.target)
    rbf_pred = rbf_search.predict(test.data)
    predictions["rbf"] = rbf_pred
    summaries.append(metrics_row("SVC-RBF", test.target, rbf_pred))
    rbf_cv = pd.DataFrame(rbf_search.cv_results_)
    rbf_cv = rbf_cv.rename(columns={"param_clf__C": "param_C", "param_clf__gamma": "param_gamma"})
    rbf_cv.to_csv(
        RESULTS / "rbf_cv_results.csv", index=False, encoding="utf-8-sig"
    )
    plot_rbf_grid(rbf_cv)

    summary = pd.DataFrame(summaries)
    summary.to_csv(
        RESULTS / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_model_comparison(summary.iloc[1:])

    linear_report = pd.DataFrame(
        classification_report(
            test.target,
            linear_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    linear_report.to_csv(
        RESULTS / "linear_classification_report.csv", encoding="utf-8-sig"
    )

    rbf_report = pd.DataFrame(
        classification_report(
            test.target,
            rbf_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    rbf_report.to_csv(
        RESULTS / "rbf_classification_report.csv", encoding="utf-8-sig"
    )

    plot_confusion(
        test.target,
        linear_pred,
        target_names,
        "linear_confusion_matrix.png",
        "Linear SVM Confusion Matrix",
    )
    plot_confusion(
        test.target,
        rbf_pred,
        target_names,
        "rbf_confusion_matrix.png",
        "RBF SVM Confusion Matrix",
    )

    linear_model = linear_search.best_estimator_.named_steps["clf"]
    top_features = top_features_per_class(
        feature_names, linear_model.coef_, target_names
    )
    top_features.to_csv(
        RESULTS / "linear_top_features.csv", index=False, encoding="utf-8-sig"
    )

    linear_decision = linear_search.decision_function(test.data)
    errors = error_analysis(
        test.data,
        test.target,
        linear_pred,
        linear_decision,
        target_names,
    )
    errors.to_csv(
        RESULTS / "linear_error_analysis.csv", index=False, encoding="utf-8-sig"
    )

    print("Generating two-dimensional decision-boundary experiment...")
    boundary_summary = plot_decision_boundaries()

    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": "20 Newsgroups, six categories",
        "categories": CATEGORIES,
        "train_size": int(len(train.data)),
        "test_size": int(len(test.data)),
        "tfidf_features": int(x_train.shape[1]),
        "linear_best_C": float(linear_search.best_params_["clf__C"]),
        "linear_cv_macro_f1": float(linear_search.best_score_),
        "rbf_best_C": float(rbf_search.best_params_["clf__C"]),
        "rbf_best_gamma": float(rbf_search.best_params_["clf__gamma"]),
        "rbf_cv_macro_f1": float(rbf_search.best_score_),
        "model_summary": summary.to_dict(orient="records"),
        "boundary_experiment": boundary_summary,
        "runtime_seconds": float(elapsed),
    }
    save_json(RESULTS / "metrics.json", metrics)

    log_summary = summary.rename(
        columns={
            "模型": "Model",
            "准确率": "Accuracy",
            "宏平均精确率": "Macro Precision",
            "宏平均召回率": "Macro Recall",
            "宏平均F1": "Macro F1",
            "加权F1": "Weighted F1",
        }
    )
    lines = [
        "News topic classification experiment",
        "------------------------------------",
        f"Training documents: {metrics['train_size']}",
        f"Test documents: {metrics['test_size']}",
        f"TF-IDF features: {metrics['tfidf_features']}",
        f"Linear SVM: C={metrics['linear_best_C']:g}, "
        f"CV macro-F1={metrics['linear_cv_macro_f1']:.4f}",
        f"RBF SVM: C={metrics['rbf_best_C']:g}, "
        f"gamma={metrics['rbf_best_gamma']:g}, "
        f"CV macro-F1={metrics['rbf_cv_macro_f1']:.4f}",
        "",
        "Test metrics",
        log_summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        "",
        f"Elapsed time: {elapsed:.1f} seconds",
    ]
    (RESULTS / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
