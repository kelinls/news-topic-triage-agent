"""Focused checks for cross-validation preprocessing isolation."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import experiment


class CrossValidationIsolationTests(unittest.TestCase):
    def test_vectorizer_fits_only_each_training_fold(self):
        self.assertTrue(
            hasattr(experiment, "build_search"),
            "Parameter search must include text preprocessing inside each fold",
        )
        fit_sizes = []

        class RecordingVectorizer(TfidfVectorizer):
            def fit_transform(self, raw_documents, y=None):
                fit_sizes.append(len(raw_documents))
                return super().fit_transform(raw_documents, y)

        docs = ["graphics image computer file"] * 10 + ["baseball team game player"] * 10
        labels = np.array([0] * 10 + [1] * 10)
        with patch.object(experiment, "build_vectorizer", RecordingVectorizer):
            search = experiment.build_search(LinearSVC(), {"C": [1.0]})
            search.n_jobs = 1
            self.assertIsInstance(search.estimator, Pipeline)
            search.fit(docs, labels)
        self.assertEqual(fit_sizes, [16] * 5 + [20])
        self.assertEqual(search.best_params_, {"clf__C": 1.0})

    def test_projection_is_inside_cross_validation_pipeline(self):
        self.assertTrue(hasattr(experiment, "build_search"))
        search = experiment.build_search(LinearSVC(), {"C": [1.0]}, project=True)
        self.assertEqual(list(search.estimator.named_steps), ["tfidf", "pca", "clf"])


if __name__ == "__main__":
    unittest.main()
