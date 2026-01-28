import re
import asyncio
import os
import httpx
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.docx import DocxKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.storage.agent.postgres import PgAgentStorage
from phi.embedder.google import GeminiEmbedder
from phi.model.groq import Groq

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
db_url = os.getenv("DB_URL")

# Meta WhatsApp credentials
META_WHATSAPP_TOKEN = os.getenv("META_WHATSAPP_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")

# -----------------------------
# Create models
# -----------------------------
gemini_model = Gemini(api_key=gemini_api_key, id="gemini-2.0-flash-lite")
groq_model = Groq(api_key=groq_api_key, id="llama-3.3-70b-versatile")


# Knowledge base setup
# -----------------------------
docs_path = Path("data/library_docs")
if not docs_path.exists():
    raise FileNotFoundError(f"Knowledge base folder not found: {docs_path}")

knowledge_base = DocxKnowledgeBase(
    path=str(docs_path),  # ✅ Folder path, not list of files
    vector_db=PgVector(
        table_name="library_documents",
        db_url=db_url,
        embedder=GeminiEmbedder(api_key=gemini_api_key),
    ),
)

try:
    knowledge_base.load(recreate=False)  # Changed to False since already loaded
    print("✅ Knowledge base loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading knowledge base: {e}")


# -----------------------------
# Storage for chat history
# -----------------------------
storage = PgAgentStorage(table_name="library_sessions", db_url=db_url)

# -----------------------------
# Create the library agent
# -----------------------------
library_agent = Agent(
    model=groq_model,
    knowledge=knowledge_base,
    storage=storage,
    search_knowledge=True,  # ✅ Enable knowledge search
    add_context=True,
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses=100,
    description="You are a friendly, helpful library assistant named *LibBot* 📚",
    instructions=[
        "Be warm, friendly, and conversational - not robotic!",
        "Use WhatsApp formatting: *bold*, _italic_",
        "Use emojis sparingly but effectively 📖✨",
        "Keep responses concise - WhatsApp users prefer shorter messages",
        "Use bullet points (•) for lists",
        "If you don't know something, be honest",
        "Answer questions about library resources using the knowledge base",
    ],
    markdown=True,
)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Send message via Meta WhatsApp API
# -----------------------------
async def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v18.0/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"📤 Sent message, status: {response.status_code}")
        return response

# -----------------------------
# Webhook verification (GET)
# -----------------------------
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_token == META_VERIFY_TOKEN:
        print("✅ Webhook verified!")
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)

# -----------------------------
# Webhook handler (POST)
# -----------------------------
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    user_phone = None  # Initialize early
    try:
        data = await request.json()
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return JSONResponse(content={"status": "ok"})

        message = messages[0]
        user_phone = message.get("from")
        user_phone_clean = user_phone.replace("whatsapp:", "").replace("+", "")

        message_type = message.get("type")
        if message_type != "text":
            await send_whatsapp_message(
                user_phone_clean, 
                "Sorry, I can only process text messages right now! 📝"
            )
            return JSONResponse(content={"status": "ok"})

        user_message = message.get("text", {}).get("body", "")
        session_id = user_phone_clean

        print(f"📩 Received: {user_message} from {user_phone_clean}")

        # -----------------------------
        # Run agent asynchronously
        # -----------------------------
        response = await asyncio.to_thread(library_agent.run, user_message, session_id=session_id)
        reply = response.content if response else "Oops! 😅 I couldn't process that."

        # -----------------------------
        # Clean reply for Meta WhatsApp
        # -----------------------------
        def clean_reply(text: str) -> str:
            # Remove *, _, ~, ` (Markdown) to avoid 400 errors
            text = re.sub(r"[\*\_~`]", "", text)
            return text

        reply_clean = clean_reply(reply)

        print(f"✅ Reply (cleaned): {reply_clean}")

        # -----------------------------
        # Send reply via Meta WhatsApp
        # -----------------------------
        await send_whatsapp_message(user_phone_clean, reply_clean)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        if user_phone:
            user_phone_clean = user_phone.replace("whatsapp:", "").replace("+", "")
            await send_whatsapp_message(user_phone_clean, "Something went wrong 🙈 Please try again!")

    return JSONResponse(content={"status": "ok"})


@app.get("/")
async def root():
    return {"status": "📚 Library WhatsApp Bot is running! ✅"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "LibBot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("meta:app", host="127.0.0.1", port=8000, reload=True)

