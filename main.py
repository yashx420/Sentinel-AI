from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CRISIS = [
    "suicide", "kill myself", "want to die", 
    "end it all", "hurt myself"
]

class UserMessage(BaseModel):
    message: str
    history: list = []


def detect_emotion(text: str):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "Classify the user's emotion in ONE word. Options: sadness, joy, anger, fear, neutral, stress, overwhelm, loneliness."},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content.lower().strip()


def check_crisis(text):
    text = text.lower()
    return any(word in text for word in CRISIS)


def create_prompt(user_message, emotion, history):
    return f"""
You are a warm, emotionally supportive companion.
User feels: {emotion}.
Respond with empathy, kindness, and validation.
Do NOT give medical advice or diagnoses.
Conversation history: {history}
User message: {user_message}
"""


@app.post("/chat")
def chat(payload: UserMessage):

    user_message = payload.message
    history = payload.history

    if check_crisis(user_message):
        return {
            "emotion": "crisis",
            "reply":
            "I'm really sorry you're feeling this way. "
            "You deserve immediate support. If you're in danger or think you might hurt yourself, "
            "please contact your local emergency number or a crisis hotline right now. "
            "You are not alone."
        }

    emotion = detect_emotion(user_message)
    prompt = create_prompt(user_message, emotion, history)

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=200
    )

    reply = completion.choices[0].message.content

    return {
        "emotion": emotion,
        "reply": reply
    }
