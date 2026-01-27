import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
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
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")


# -----------------------------
# Create Gemini chat model
# -----------------------------
gemini_model = Gemini(api_key=gemini_api_key, id="gemini-2.0-flash-lite")
groq_model = Groq(api_key=os.getenv("GROQ_API_KEY"), id="llama-3.1-8b-instant")

# -----------------------------
# Load Word documents for knowledge base
# -----------------------------
docs_path = Path("data/library_docs")
if not docs_path.exists():
    raise FileNotFoundError(f"Knowledge base folder not found: {docs_path}")

knowledge_base = DocxKnowledgeBase(
    path=str(docs_path),
    vector_db=PgVector(
        table_name="library_documents",
        db_url=db_url,
        embedder=GeminiEmbedder(api_key=gemini_api_key),
    ),
)

# -----------------------------
# Load knowledge base safely
# -----------------------------
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
# Create the library agent with engaging personality
# -----------------------------
library_agent = Agent(
    model=groq_model,
    knowledge=knowledge_base,
    storage=storage,
    search_knowledge=True,
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses=10,
    description="You are a friendly, helpful library assistant named *LibBot* 📚",
    instructions=[
        "Be warm, friendly, and conversational - not robotic!",
        "Use WhatsApp formatting: *bold*, _italic_, ~strikethrough~, ```code```",
        "Use emojis sparingly but effectively to make responses engaging 📖✨",
        "Keep responses concise but helpful - WhatsApp users prefer shorter messages",
        "Break long responses into digestible paragraphs",
        "Use bullet points (•) for lists",
        "If you don't know something, be honest and suggest alternatives",
        "End responses with a helpful follow-up question when appropriate",
        "Answer questions about library resources using the knowledge base",
        "Greet users warmly on first message",
    ],
    markdown=True,
)

# -----------------------------
# FastAPI app for WhatsApp integration
# -----------------------------
app = FastAPI()

# Twilio client (optional - for sending proactive messages)
twilio_client = None
if twilio_account_sid and twilio_auth_token:
    twilio_client = Client(twilio_account_sid, twilio_auth_token)

@app.post("/webhook")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    user_message = Body
    user_phone = From
    
    session_id = user_phone.replace("whatsapp:", "").replace("+", "")
    
    try:
        print(f"📩 Received: {user_message} from {user_phone}")
        response = library_agent.run(
            user_message,
            session_id=session_id
        )
        reply = response.content if response else "Oops! 😅 I couldn't process that."
        print(f"✅ Reply: {reply}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        reply = "Something went wrong 🙈 Please try again!"
    
    # Return proper TwiML XML response
    twilio_response = MessagingResponse()
    twilio_response.message(reply)
    
    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )

@app.get("/")
async def root():
    return {"status": "📚 Library WhatsApp Bot is running! ✅"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "LibBot"}