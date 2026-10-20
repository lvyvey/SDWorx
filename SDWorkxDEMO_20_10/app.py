"""AskAround demo server: a Teams-style pop-up on top of the knowledge graph.

Run:  .venv/bin/python app.py   then open http://localhost:5050
"""

from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, render_template, request

from askaround import create_demo

app = Flask(__name__)
_lock = threading.Lock()
engine = create_demo()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    with _lock:
        return jsonify(engine.ask(question))


@app.post("/api/rate")
def rate():
    body = request.json or {}
    with _lock:
        answer = engine.rate_answer(body["answer_id"], bool(body["helpful"]))
    return jsonify({"helpful": answer.helpful, "not_helpful": answer.not_helpful})


@app.post("/api/verify")
def verify():
    """Demo shortcut for the expert's reply to the 'please verify' message."""
    body = request.json or {}
    with _lock:
        answer = engine.verify(body["question"], body["answer_text"], body["employee_id"],
                               body.get("document_ids", []), bool(body["correct"]), body.get("answer_id"))
    return jsonify({"answer_id": answer.id if answer else None})


@app.post("/api/answer")
def save_answer():
    """Demo shortcut: store the answer an expert gave after being contacted."""
    body = request.json or {}
    text = body.get("answer_text", "").strip()
    if not text:
        return jsonify({"error": "The answer is empty."}), 400
    with _lock:
        answer = engine.save_answer(body["question"], text, body["employee_id"], body.get("document_ids", []))
    return jsonify({"answer_id": answer.id})


@app.post("/api/reset")
def reset():
    global engine
    with _lock:
        engine = create_demo()
    return jsonify({"ok": True})


@app.get("/api/graph")
def graph():
    with _lock:
        return jsonify(engine.graph.to_json())


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", "5050")))
