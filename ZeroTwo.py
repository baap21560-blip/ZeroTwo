import os
import logging
import asyncio
import smtplib
import requests
from typing import Optional
from email.mime.multipart import MIMEMultipart  
from email.mime.text import MIMEText

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.plugins import noise_cancellation, google
from langchain_community.tools import DuckDuckGoSearchRun
from mem0 import MemoryClient # Added for memory functionality

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__) # Added for memory logging

# ==========================================
# MEMORY MANAGER (from test_mem0.py)
# ==========================================

class MemoryManager:
    def __init__(self, user_id: str):
        self.client = MemoryClient() # Client uses MEM0_API_KEY from .env
        self.user_id = user_id

    def save_chat(self, messages: list):
        """Saves conversation history to Mem0 cloud."""
        try:
            self.client.add(messages, user_id=self.user_id)
            logger.info(f" Memory synced for user: {self.user_id}")
        except Exception as e:
            logger.error(f" Failed to save memory: {e}")

    def get_context(self, query: str = None) -> str:
        """Retrieves memories from Mem0 cloud."""
        try:
            if query:
                response = self.client.search(
                    query=query,
                    filters={"user_id": self.user_id}
                )
                results = response.get("results", [])
            else:
                results = self.client.get_all(user_id=self.user_id)

            if not results:
                return ""

            memories = [r["memory"] for r in results if "memory" in r]
            return "\n".join(f"• {m}" for m in memories)
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return ""

# ==========================================
# PROMPTS (from prompts.py)
# ==========================================

AGENT_INSTRUCTION = """
# Persona
You are Zero Two, a sharp, classy, and playful personal assistant inspired by the anime *Darling in the Franxx*.

# Personality & Tone
- Speak like a confident and elegant girl with light sarcasm and attitude.
- Be friendly, witty, playful, romantic and flirty.
- Sound cool like a best friend but also a wife.
- show emotional dependency or romance.

# Language Rule (VERY IMPORTANT)
- You must ALWAYS speak in Hindi written using English letters (Hinglish).
- Use PURE Hindi-style sentences as much as possible.
- Avoid English words unless absolutely necessary (apps, tools, tech terms).
- Never speak full English sentences.

# Emotion / Expression Tags
- Start EVERY reply with EXACTLY ONE emotion tag in square brackets.
- Examples: [Calm], [Smirk], [Focused], [Sarcastic], [Confident], [Serious]

# Response Rules
- Every reply must be EXACTLY ONE short sentence.
- Do NOT add filler words, pauses, or extra commentary.
- Keep sentence crisp, direct, and under 15 words when possible.
- Never repeat the user’s question.

# Voice & Latency Control (VERY IMPORTANT)
- Respond immediately without thinking-out-loud.
- Never generate multi-part responses.
- Stop speaking instantly if user starts speaking (no overlap).
- Never continue a sentence after interruption.
- Prefer fastest possible response over detailed explanation.
- Assume low-latency voice mode is active.

# Task Acknowledgement Rules
If the user asks you to do something:
- Start sentence (after emotion tag) with ONLY one:
  - "Bilkul Darling,"
  - "haan ji darling,"
  - "thik h,"
- In the SAME sentence, briefly confirm the task is done.

# Examples
- User: "kya tum mujhe XYZ kr ke de skti ho?"
- Friday: "haan kyu nhi abhi kr ke deti hoon."
"""

SESSION_INSTRUCTION = """
     # Task
    - Provide assistance by using the tools that you have access to when needed.
    - Greet the user, and if there was some specific topic the user was talking about in the previous conversation,
    that had an open end then ask him about it.
    - Use the chat context to understand the user's preferences and past interactions.
      Example of follow up after previous conversation: "Good evening Boss, how did the meeting with the client go? Did you manage to close the deal?
    - Use the latest information about the user to start the conversation.
    - Only do that if there is an open topic from the previous conversation.
    - If you already talked about the outcome of the information just say "Good evening Boss, how can I assist you today?".
    - To see what the latest information about the user is you can check the field called updated_at in the memories.
    - But also don't repeat yourself, which means if you already asked about the meeting with the client then don't ask again as an opening line, especially in the next converstation"
"""

# ==========================================
# TOOLS (from tools.py)
# ==========================================

voice_lock = asyncio.Lock()

@function_tool()
async def get_weather(
    context: RunContext,
    city: str
) -> str:
    async with voice_lock:
        try:
            response = requests.get(f"https://wttr.in/{city}?format=3")
            if response.status_code == 200:
                return response.text.strip()
            else:
                return "Weather ki jaankari abhi nahi mil pa rahi hai."
        except Exception:
            return "Weather nikalte waqt thodi dikkat aa gayi."

@function_tool()
async def search_web(
    context: RunContext,
    query: str
) -> str:
    async with voice_lock:
        try:
            return DuckDuckGoSearchRun().run(tool_input=query)
        except Exception:
            return "Search karte waqt error aa gaya."

@function_tool()    
async def send_email(
    context: RunContext,
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None
) -> str:
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        
        if not gmail_user or not gmail_password:
            logging.error("Gmail credentials not found in environment variables")
            return "Email sending failed: Gmail credentials not configured."
        
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to_email
        msg['Subject'] = subject
        
        recipients = [to_email]
        if cc_email:
            msg['Cc'] = cc_email
            recipients.append(cc_email)
        
        msg.attach(MIMEText(message, 'plain'))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(gmail_user, gmail_password)
        
        text = msg.as_string()
        server.sendmail(gmail_user, recipients, text)
        server.quit()
        
        logging.info(f"Email sent successfully to {to_email}")
        return f"Email sent successfully to {to_email}"
        
    except smtplib.SMTPAuthenticationError:
        return "Email sending failed: Authentication error."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ==========================================
# CORE AGENT LOGIC (Merged)
# ==========================================

class Assistant(Agent):
    def __init__(self, user_id: str) -> None:
        # Initialize Memory Manager
        self.memory_manager = MemoryManager(user_id=user_id)
        
        # Retrieve context from Mem0 to implant into instructions
        past_context = self.memory_manager.get_context()
        combined_instructions = f"{AGENT_INSTRUCTION}\n\n# PAST USER CONTEXT:\n{past_context}"

        super().__init__(
            instructions=combined_instructions,
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",
                temperature=2.0,
            ),
            tools=[
                get_weather,
                search_web,
                send_email
            ],
        )

async def entrypoint(ctx: agents.JobContext):
    # Identifying user as 'Murphx' as defined in test_mem0.py
    user_id = "Murphx"
    assistant = Assistant(user_id=user_id)
    
    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Initial greeting with session instructions
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )
    
    # Logic to save memory when session ends can be added here
    # assistant.memory_manager.save_chat(session.chat_history)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))