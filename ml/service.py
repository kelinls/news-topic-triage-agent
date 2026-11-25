"""Loopback inference API for the Java AgentX tools."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import joblib
import numpy as np

from .classifier import cosine_similarities, neighbor_distribution, softmax


MODEL = Path(__file__).resolve().parent / "artifacts" / "news_triage_agent.joblib"
MAX_REQUEST_BYTES = 1024 * 1024


def classify(model, text: str) -> dict:
    matrix = model.vectorizer.transform([text])
    scores = np.asarray(model.classifier.decision_function(matrix)).reshape(1, -1)
    probability = softmax(scores, model.policy.temperature)[0]
    first, second = np.argsort(probability)[-2:][::-1]
    return {
        "label": model.target_names[first],
        "confidence": round(float(probability[first]), 6),
        "margin": round(float(probability[first] - probability[second]), 6),
        "features": [row["feature"] for row in model._top_features(matrix, first)],
    }


def retrieve(model, text: str, top_k: int) -> dict:
    matrix = model.vectorizer.transform([text])
    similarity = cosine_similarities(matrix, model.train_matrix)
    distribution, _ = neighbor_distribution(similarity, model.train_labels, len(model.target_names), top_k)
    winner = int(np.argmax(distribution))
    return {
        "label": model.target_names[winner],
        "agreement": round(float(distribution[winner]), 6),
        "distribution": {
            label: round(float(share), 6)
            for label, share in zip(model.target_names, distribution, strict=True)
            if share > 0
        },
    }


def handler(model):
    class InferenceHandler(BaseHTTPRequestHandler):
        def response(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.response(200, {"status": "ok"})
            else:
                self.response(404, {"error": "Unknown route"})

        def do_POST(self) -> None:
            if self.path not in ("/classify", "/retrieve", "/baseline"):
                self.response(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or not isinstance(request.get("text"), str):
                    raise ValueError("Text must be a string")
                text = request["text"].replace("\x00", " ").strip()
                if self.path == "/baseline":
                    fixed = model.triage(text)
                    result = {"status": fixed["status"], "label": fixed["suggested_label"]}
                else:
                    if not 40 <= len(text) <= 50_000:
                        raise ValueError("Input must contain 40 to 50000 characters")
                    if self.path == "/classify":
                        result = classify(model, text)
                    else:
                        top_k = request.get("topK")
                        if type(top_k) is not int or not 3 <= top_k <= 15:
                            raise ValueError("topK must be an integer between 3 and 15")
                        result = retrieve(model, text, top_k)
                self.response(200, result)
            except ValueError as exc:
                self.response(400, {"error": str(exc)})

    return InferenceHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    model = joblib.load(MODEL)["agent"]
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler(model)) as server:
        print(f"ML service on 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
