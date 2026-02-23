from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(body: CommentRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Analyze the sentiment of this comment and respond with ONLY a JSON object.
Comment: "{body.comment}"
Rules:
- sentiment must be exactly one of: "positive", "negative", "neutral"
- rating must be an integer 1-5 (5=highly positive, 1=highly negative, 3=neutral)
Respond with only this JSON format, nothing else:
{{"sentiment": "positive", "rating": 5}}"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return {"sentiment": result["sentiment"], "rating": result["rating"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
