"""Template for local_settings.py (the real file is git-ignored).

    cp local_settings.example.py local_settings.py && chmod 600 local_settings.py

Plain constants only: orange_harness.py runs this file to read them, so put
no other code here. Never commit local_settings.py.
"""

# WALLX VLA WebSocket endpoint (local station model).
WALLX_ENDPOINT = "ws://192.168.1.10:33677"  # plain ws: this server does not speak TLS
# Handshake headers for WALLX, e.g. {"Authorization": "Bearer ..."}. Empty for a local WALLX.
WALLX_HEADERS = {}

# Vision LLM: any OpenAI Chat Completions-compatible endpoint. For Gemini, use the
# OpenAI-compatible URL below, NOT the Interactions or generateContent API.
AGENT_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
AGENT_MODEL = "gemini-3.5-flash-lite"  # ~1 s/verify. "gemini-3.5-flash" measured 1.5-14 s (load-dependent) and hits the 10 s timeout
AGENT_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

# Gemini thinking level ("minimal" keeps verifies fast; on gemini-3.5-flash default thinking
# measured 4-14 s, past the 10 s timeout). None = model default.
AGENT_REASONING_EFFORT = "minimal"
