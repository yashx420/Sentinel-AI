# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import openai

openai.api_key = "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"

# Create API app
app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion model once
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

CRISIS = [
    "suicide", "kill myself", "want to die", 
    "end it all", "hurt myself"
]

# Request body
class UserMessage(BaseModel):
    message: str
    history: list = []  # optional chat history


def check_crisis(text):
    text = text.lower()
    return any(word in text for word in CRISIS)


def create_empathetic_prompt(user_message, emotion, history):
    return f"""
You are an emotionally supportive companion.
The user feels: {emotion}.
Always respond with warmth, empathy, and validation.
Do NOT give medical advice or diagnostic claims.
Avoid telling the user what to do.
Avoid clinical language.

Here is the message:
"{user_message}"

Conversation history:
{history}
"""


@app.post("/chat")
def chat(payload: UserMessage):

    user_message = payload.message
    history = payload.history

    # Crisis handling
    if check_crisis(user_message):
        return {
            "emotion": "crisis",
            "reply":
            "I'm really sorry you’re feeling this way. "
            "You deserve immediate support. If you're in danger or think you might hurt yourself, "
            "contact your local emergency number or a crisis hotline right now. "
            "You are not alone."
        }

    # Emotion detection
    emotion = emotion_model(user_message)[0]["label"]

    # LLM response
    prompt = create_empathetic_prompt(user_message, emotion, history)

    completion = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200
    )

    reply = completion.choices[0].message.content

    return {
        "emotion": emotion,
        "reply": reply
    }
