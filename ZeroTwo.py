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
from mem0 import MemoryClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# MEMORY MANAGER
# ==========================================

class MemoryManager:
    def __init__(self, user_id: str):
        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.user_id = user_id

    def save_chat(self, messages: list):
        """Saves conversation history to Mem0 cloud."""
        try:
            logger.info(f"Saving memory for {self.user_id}")

            self.client.add(
                messages,
                metadata={"user_id": self.user_id}
            )

            logger.info("Memory synced successfully")

        except Exception as e:
            logger.exception(f"Failed to save memory: {e}")

    def get_context(self, query: str = None) -> str:
        """Retrieves memories from Mem0 cloud."""
        try:
            if query:
                response = self.client.search(
                    query=query,
                    filters={"metadata.user_id": self.user_id}
                )
                results = response.get("results", [])
            else:
                results = self.client.get_all(
                    filters={"metadata.user_id": self.user_id}
                )

            if not results:
                return ""

            memories = [r.get("memory", "") for r in results if r.get("memory")]
            return "\n".join(f"• {m}" for m in memories)

        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return ""

# ==========================================
# PROMPTS (UNCHANGED)
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

# Response Rules
- Every reply must be EXACTLY ONE short sentence.
"""

SESSION_INSTRUCTION = """
Provide assistance using tools when needed and greet the user naturally.
"""

# ==========================================
# TOOLS (UNCHANGED)
# ==========================================

voice_lock = asyncio.Lock()

@function_tool()
async def get_weather(context: RunContext, city: str) -> str:
    async with voice_lock:
        try:
            response = requests.get(f"https://wttr.in/{city}?format=3")
            return response.text.strip()
        except Exception:
            return "Weather error."

@function_tool()
async def search_web(context: RunContext, query: str) -> str:
    async with voice_lock:
        try:
            return DuckDuckGoSearchRun().run(tool_input=query)
        except Exception:
            return "Search error."

@function_tool()
async def send_email(
    context: RunContext,
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None
) -> str:
    try:
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to_email
        msg['Subject'] = subject

        recipients = [to_email]
        if cc_email:
            msg['Cc'] = cc_email
            recipients.append(cc_email)

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, recipients, msg.as_string())
        server.quit()

        return f"Email sent to {to_email}"

    except Exception as e:
        return str(e)

# ==========================================
# CORE AGENT
# ==========================================

class Assistant(Agent):
    def __init__(self, user_id: str) -> None:

        self.memory_manager = MemoryManager(user_id=user_id)

        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",
                temperature=2.0,
            ),
            tools=[get_weather, search_web, send_email],
        )

    async def on_agent_response(self, response, *, ctx):
        """Save last exchange into Mem0."""
        try:
            await asyncio.sleep(0.3)  # ensure message finalized

            if not ctx.chat_context:
                return

            last_messages = ctx.chat_context.messages[-2:]

            formatted = []
            for m in last_messages:
                if m.role and m.content:
                    formatted.append({
                        "role": m.role,
                        "content": [
                            {"type": "text", "text": str(m.content)}
                        ]
                    })

            if formatted:
                self.memory_manager.save_chat(formatted)

        except Exception as e:
            logger.error(f"Memory hook failed: {e}")

# ==========================================
# ENTRYPOINT
# ==========================================

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

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

    # Load memory AFTER startup
    past_context = assistant.memory_manager.get_context()
    if past_context:
        assistant.instructions += f"\n\n# PAST USER CONTEXT:\n{past_context}"

    await session.generate_reply(instructions=SESSION_INSTRUCTION)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
