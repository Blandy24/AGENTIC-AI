import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.docx import DocxKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.storage.agent.postgres import PgAgentStorage

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
db_url = os.getenv("DB_URL")

# Create Gemini chat model
gemini_model = Gemini(api_key=gemini_api_key, id="gemini-1.5-flash")


# Knowledge base from Word documents (.doc and .docx)
knowledge_base = DocxKnowledgeBase(
    path="data/library_docs",  # Put all your Word files here
    vector_db=PgVector(
        table_name="library_documents",
        db_url=db_url,
    ),
)

# Load knowledge base (run once, then set recreate=False)
knowledge_base.load(recreate=False)

# Storage for chat history
storage = PgAgentStorage(table_name="library_sessions", db_url=db_url)

# Create the library agent
library_agent = Agent(
    model=gemini_model,
    knowledge=knowledge_base,
    storage=storage,
    search_knowledge=True,
    read_chat_history=True,
    add_history_to_messages=True,
    num_history_responses=10,  # Adjust as needed
    description="You are a helpful library assistant.",
    instructions=["Answer questions about library resources using the knowledge base."],
    show_tool_calls=True,
    markdown=True,
)

# FastAPI app
app = FastAPI()

class WhatsAppMessage(BaseModel):
    user_message: str
    session_id: str  # Use WhatsApp user ID/phone number

@app.post("/webhook")
async def whatsapp_webhook(message: WhatsAppMessage):
    response = library_agent.run(
        message.user_message,
        session_id=message.session_id
    )
    return {"reply": response.content}


