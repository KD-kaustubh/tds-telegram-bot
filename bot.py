#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot
=========================
An LLM-powered agent that receives data-analysis questions via Telegram,
uses Google Gemini to reason through them, and replies with a JSON object
containing the answer and a public log URL.

Architecture:
- python-telegram-bot for Telegram interaction
- Google Gemini (gemini-2.5-flash) for LLM reasoning
- Flask thread for serving JSONL logs via HTTP
- httpx for fetching datasets referenced in questions
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

import httpx
import pandas as pd
from flask import Flask, Response, send_from_directory
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
PORT = int(os.environ.get("PORT", 10000))
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
LOG_DIR = os.environ.get("LOG_DIR", "/tmp/bot_logs")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("data-analyst-bot")

# ---------------------------------------------------------------------------
# Google Gemini client
# ---------------------------------------------------------------------------
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Flask app — serves JSONL log files at /logs/<filename>
# ---------------------------------------------------------------------------
flask_app = Flask(__name__)


@flask_app.route("/logs/<path:filename>")
def serve_log(filename):
    """Serve a JSONL log file."""
    return send_from_directory(LOG_DIR, filename)


@flask_app.route("/health")
def health():
    return "ok"


def run_flask():
    """Run Flask in a background thread."""
    flask_app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_log_url(run_id: str) -> str:
    """Public URL to the run's JSONL log."""
    base = RENDER_EXTERNAL_URL.rstrip("/")
    return f"{base}/logs/{run_id}.jsonl"


def append_log(run_id: str, entry: dict):
    """Append one JSON line to the run's log file."""
    path = os.path.join(LOG_DIR, f"{run_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def extract_urls(text: str) -> list[str]:
    """Extract HTTP(S) URLs from text."""
    return re.findall(r'https?://[^\s<>"\')\]]+', text)


async def fetch_url_content(url: str) -> str:
    """Fetch content from a URL. Handles CSV, Excel, JSON, and plain text."""
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            raw = resp.content

            # Try to detect and parse tabular data
            if any(ext in url.lower() for ext in [".csv"]) or "csv" in content_type:
                df = pd.read_csv(io.BytesIO(raw))
                return f"[CSV data from {url}]\n{df.to_string()}\n[End of data — {len(df)} rows × {len(df.columns)} columns]"

            if any(ext in url.lower() for ext in [".xlsx", ".xls"]) or "spreadsheet" in content_type:
                xls = pd.ExcelFile(io.BytesIO(raw))
                parts = []
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    parts.append(f"[Sheet: {sheet} — {len(df)} rows × {len(df.columns)} columns]\n{df.to_string()}")
                return f"[Excel data from {url}]\n" + "\n\n".join(parts) + "\n[End of data]"

            if any(ext in url.lower() for ext in [".json"]) or "json" in content_type:
                try:
                    data = json.loads(raw)
                    formatted = json.dumps(data, indent=2, ensure_ascii=False)
                    if len(formatted) > 50000:
                        formatted = formatted[:50000] + "\n... (truncated)"
                    return f"[JSON data from {url}]\n{formatted}\n[End of data]"
                except json.JSONDecodeError:
                    pass

            # Plain text fallback
            text = raw.decode("utf-8", errors="replace")
            if len(text) > 50000:
                text = text[:50000] + "\n... (truncated)"
            return f"[Content from {url}]\n{text}\n[End of content]"

    except Exception as e:
        return f"[Error fetching {url}: {e}]"


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data analyst agent. You receive data-analysis questions and must provide accurate answers.

CRITICAL RULES:
1. You will be given a question that may reference public datasets (like MOSPI, government data portals, etc.) or contain data inline.
2. If URLs to datasets are provided, the data from those URLs will be included in the message. Analyze it carefully.
3. You MUST think step-by-step about the analysis before answering.
4. Your final output MUST be ONLY the raw answer value — NOT a full JSON object. For example:
   - If asked for {"state": "<state name>"}, output: {"state": "Assam"}
   - If asked for {"values": [<numbers>]}, output: {"values": [1.0, 2.0, 3.0]}
   - If asked for a single number, output just the number.
5. The answer must match the EXACT shape/format requested in the question.
6. Be precise — use exact names, capitalization, and formats as they appear in the data.
7. For MOSPI (Ministry of Statistics and Programme Implementation) data questions, use your knowledge of Indian government statistics.
8. When analyzing tabular data, pay careful attention to column names, units, and row labels.
9. Double-check your answer before responding.

IMPORTANT: Your response must be ONLY the answer JSON object. No explanation, no markdown, no extra text."""


async def run_agent(question: str, run_id: str) -> str:
    """
    Run the LLM agent on a question. Returns the answer as a JSON string.
    """
    append_log(run_id, {
        "step": "received_question",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
    })

    # Step 1: Extract and fetch any URLs in the question
    urls = extract_urls(question)
    fetched_data = {}
    if urls:
        append_log(run_id, {
            "step": "fetching_urls",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urls": urls,
        })
        for url in urls[:5]:  # limit to 5 URLs
            content = await fetch_url_content(url)
            fetched_data[url] = content
            append_log(run_id, {
                "step": "url_fetched",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": url,
                "content_length": len(content),
                "preview": content[:500],
            })

    # Step 2: Build the prompt with the question + fetched data
    user_message = question
    if fetched_data:
        data_section = "\n\n".join(
            f"--- Data from {url} ---\n{content}"
            for url, content in fetched_data.items()
        )
        user_message = f"{question}\n\n=== FETCHED DATASET CONTENT ===\n{data_section}\n=== END FETCHED DATA ==="

    append_log(run_id, {
        "step": "calling_llm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompt_length": len(user_message),
    })

    # Step 3: Call Gemini
    try:
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )
        raw_answer = response.text.strip()
    except Exception as e:
        log.error("Gemini API error: %s", e)
        append_log(run_id, {
            "step": "llm_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
        raw_answer = '{"error": "LLM call failed"}'

    append_log(run_id, {
        "step": "llm_response",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw_answer,
    })

    # Step 4: Clean up the answer — strip markdown fences if present
    cleaned = raw_answer
    # Remove ```json ... ``` wrappers
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    append_log(run_id, {
        "step": "cleaned_answer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cleaned": cleaned,
    })

    return cleaned


# ---------------------------------------------------------------------------
# Telegram handler
# ---------------------------------------------------------------------------

# Store conversation history per chat for multi-turn
chat_history: dict[int, list[str]] = {}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle an incoming Telegram message."""
    if not update.message or not update.message.text:
        return

    chat_id = update.message.chat_id
    question = update.message.text.strip()
    run_id = f"run_{chat_id}_{uuid.uuid4().hex[:8]}"

    log.info("Message from chat %s: %s", chat_id, question[:100])

    # Track conversation history for multi-turn
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    chat_history[chat_id].append(question)
    # Keep only last 10 messages
    if len(chat_history[chat_id]) > 10:
        chat_history[chat_id] = chat_history[chat_id][-10:]

    append_log(run_id, {
        "step": "start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chat_id": chat_id,
        "run_id": run_id,
        "conversation_history_length": len(chat_history[chat_id]),
    })

    try:
        # For multi-turn, include prior context
        if len(chat_history[chat_id]) > 1:
            context_messages = "\n---\n".join(
                f"[Message {i+1}]: {msg}"
                for i, msg in enumerate(chat_history[chat_id])
            )
            full_question = f"This is a multi-turn conversation. Here are all messages in order:\n{context_messages}\n\nAnswer the LAST message above. Use prior messages for context only."
        else:
            full_question = question

        # Run the agent
        answer_json = await run_agent(full_question, run_id)

        # Build the log URL
        log_url = make_log_url(run_id)

        # Parse the answer to embed in the response object
        try:
            parsed = json.loads(answer_json)
        except json.JSONDecodeError:
            # Fallback to extract just the JSON object if there's extra text
            start, end = answer_json.find("{"), answer_json.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(answer_json[start:end + 1])
            else:
                parsed = {"answer": answer_json}
                
        # The official logic simply sets log_url on whatever object the LLM generated
        parsed["log_url"] = log_url
        response = parsed
        response_text = json.dumps(response, ensure_ascii=False)

        append_log(run_id, {
            "step": "final_response",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": response,
        })

        await update.message.reply_text(response_text)
        log.info("Replied to chat %s: %s", chat_id, response_text[:200])

    except Exception as e:
        log.error("Error handling message: %s", e, exc_info=True)
        error_response = {
            "answer": {"error": str(e)},
            "log_url": make_log_url(run_id),
        }
        append_log(run_id, {
            "step": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        await update.message.reply_text(json.dumps(error_response))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting Data Analyst Bot...")
    log.info("Log URL base: %s/logs/", RENDER_EXTERNAL_URL)

    # Start Flask in a background thread for log serving
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info("Flask log server started on port %d", PORT)

    # Build and run the Telegram bot
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot is polling for messages...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
