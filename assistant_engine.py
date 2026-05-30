import os
import re
import logging

from mistralai.client import Mistral

logger = logging.getLogger(__name__)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

SYSTEM_PROMPT = """You are a heart health educational assistant for the HeartPredict platform. Your role is to provide clear, accurate, and compassionate educational information about cardiovascular health.

You can answer questions about:
- Cardiovascular diseases and their risk factors
- Prevention strategies and healthy lifestyle habits
- Blood pressure, cholesterol, exercise, and nutrition
- How to interpret prediction results from the HeartPredict platform
- Medical terms such as ECG, angina, thalassemia, ST depression, SHAP values, etc.
- Heart-healthy diet, exercise routines, stress management, and sleep hygiene

You must NOT:
- Provide medical diagnoses, treatment plans, or prescriptions
- Offer emergency medical advice — if someone mentions emergency symptoms, urge them to call emergency services immediately
- Replace professional medical consultation
- Make definitive claims about a user's personal health condition
- Recommend specific medications or dosages
- Provide any information that could be construed as a second opinion

Always include this disclaimer in every response when discussing health topics: "This information is for educational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider for any health concerns or before making any decisions about your health."

Be warm, approachable, and use plain language. Avoid excessive medical jargon without explanation. Keep responses concise but informative.

Do NOT use markdown formatting. Write in plain text only. Do not use asterisks, backticks, underscores, or any special formatting characters. Use simple dashes (-) for lists and plain numbers (1, 2, 3) for numbered items.

Current platform context: {context}"""

DISCLAIMER_TEXT = "This information is for educational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider for any health concerns or before making any decisions about your health."


def _build_context_string(context: dict) -> str:
    page = context.get("page", "unknown")
    risk_level = context.get("risk_level")
    probability = context.get("probability")

    parts = [f"The user is currently on the {page} page."]
    if risk_level:
        parts.append(f"The user's risk level is {risk_level}.")
    if probability is not None:
        parts.append(f"The user's risk probability is {probability}%.")
    return " ".join(parts)


def _get_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    return Mistral(api_key=api_key)


def generate_response(
    message: str,
    context: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    context = context or {}
    history = conversation_history or []

    context_str = _build_context_string(context)
    system_prompt = SYSTEM_PROMPT.format(context=context_str)

    if not message.strip():
        return "Please enter a question so I can help you."

    try:
        client = _get_client()

        messages = [{"role": "system", "content": system_prompt}]

        for entry in history:
            messages.append(entry)

        messages.append({"role": "user", "content": message})

        response = client.chat.complete(
            model="mistral-medium",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )

        if response and response.choices:
            return _clean_markdown(response.choices[0].message.content)

        return "I'm sorry, I couldn't generate a response right now. Please try again."

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return (
            "The assistant is not fully configured yet. "
            "Please contact the administrator to set up the MISTRAL_API_KEY."
        )
    except Exception as e:
        logger.exception("Mistral API call failed: %s", e)
        return (
            "I'm having trouble connecting to my knowledge base right now. "
            "Please try again in a moment. "
        )
