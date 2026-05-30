"""
assistant_routes.py — Flask blueprint for the Heart Health Assistant API
"""

from flask import Blueprint, jsonify, request, render_template, session
from flask_login import login_required, current_user
from assistant_engine import generate_response

assistant_bp = Blueprint("assistant", __name__, url_prefix="/assistant")

MAX_HISTORY = 20


@assistant_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    context = data.get("context") or {}

    if not message:
        return jsonify({"error": "Message is required"}), 400

    history_key = f"chat_history_{current_user.id}"
    conversation = session.get(history_key, [])

    response = generate_response(
        message,
        context=context,
        conversation_history=conversation,
    )

    conversation.append({"role": "user", "content": message})
    conversation.append({"role": "assistant", "content": response})

    if len(conversation) > MAX_HISTORY:
        conversation = conversation[-MAX_HISTORY:]

    session[history_key] = conversation

    return jsonify({"response": response})


@assistant_bp.route("/page")
@login_required
def assistant_page():
    return render_template("assistant.html")
