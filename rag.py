import os
import time
from collections import deque
from dotenv import load_dotenv
from google import genai

# LOAD & VERIFY GEMINI KEY

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# --- verify key once ---
try:
    test = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Reply with OK"
    )
    print("✅ Gemini key verified:", test.text.strip())
except Exception as e:
    raise RuntimeError(f"❌ Gemini key invalid: {e}")


# SIMPLE RAG MEMORY

rag_memory = deque(maxlen=200)
start_time = time.time()

def add_event(gaze_x, gaze_y, blink_count, emotion):
    """
    Call this from your vision model
    """
    t = time.time() - start_time
    rag_memory.append(
        f"Time {t:.1f}s | Gaze({gaze_x:.2f},{gaze_y:.2f}) "
        f"| Blinks={blink_count} | Emotion={emotion}"
    )

def ask_rag(question):
    if not rag_memory:
        return "No vision data collected yet."

    context = "\n".join(list(rag_memory)[-30:])

    prompt = f"""
You are analyzing user attention using gaze direction,
blink frequency, and facial emotion.

Context:
{context}

Question:
{question}

Answer clearly in 2–3 lines.
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text.strip()


# DEMO USAGE (REMOVE LATER)

if __name__ == "__main__":
    # Simulated important data from your model
    add_event(0.52, 0.48, 3, "Neutral")
    add_event(0.50, 0.46, 3, "Neutral")
    add_event(0.49, 0.45, 4, "Happy")

    answer = ask_rag("Is the user's gaze stable or distracted?")
    # answer = ask_rag("Is the user attentive?")
    print("\n🧠 Gemini RAG Answer:")
    print(answer)
