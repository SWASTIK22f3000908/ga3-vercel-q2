from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import math
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load telemetry data
file_path = os.path.join(os.path.dirname(__file__), "../q-vercel-latency.json")
with open(file_path) as f:
    telemetry = json.load(f)

# Initialize OpenAI (AIPipe token from Vercel env)
client = OpenAI(api_key=os.getenv("AIPIPE_TOKEN"))

# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Telemetry Analytics
# -------------------------
class TelemetryRequest(BaseModel):
    regions: list[str]
    threshold_ms: float

@app.post("/")
def analyze_latency(req: TelemetryRequest):

    result = {}

    for region in req.regions:
        region_data = [r for r in telemetry if r["region"] == region]

        if not region_data:
            continue

        latencies = [r["latency_ms"] for r in region_data]
        uptimes = [r["uptime_pct"] for r in region_data]

        avg_latency = sum(latencies) / len(latencies)

        sorted_lat = sorted(latencies)
        index_95 = math.ceil(0.95 * len(sorted_lat)) - 1
        p95_latency = sorted_lat[index_95]

        avg_uptime = sum(uptimes) / len(uptimes)

        breaches = len([l for l in latencies if l > req.threshold_ms])

        result[region] = {
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "avg_uptime": avg_uptime,
            "breaches": breaches
        }

    return result

# -------------------------
# Sentiment Analysis
# -------------------------
class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(req: CommentRequest):

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Classify sentiment strictly into positive, negative, neutral and rate intensity 1-5:\n\n{req.comment}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "sentiment_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))