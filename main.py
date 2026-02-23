from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment")
async def analyze_comment(body: CommentRequest):
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Classify sentiment as exactly 'positive', 'negative', or 'neutral'. Rate 1-5 where 5=highly positive, 1=highly negative, 3=neutral."
                },
                {"role": "user", "content": body.comment}
            ],
            response_format=SentimentResponse,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
