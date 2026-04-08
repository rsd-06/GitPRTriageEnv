from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.environment import DevTriageEnvironment
from models import TriageAction

app = FastAPI(title="GitPRTriage Env")

@app.get("/")
def root():
    # Provide a friendly landing message instead of FastAPI's default 404
    return {"message": "GitPRTriage Env API is live! Navigate to /docs to use the interactive Swagger UI.", "status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DevTriageEnvironment()

@app.get("/health")
def health():
    return {"status": "healthy", "issues_loaded": len(env.all_issues)}

@app.post("/reset")
def reset():
    return env.reset().model_dump()

@app.post("/step")
def step(action: TriageAction):
    return env.step(action).model_dump()

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/tasks")
def tasks():
    return [
        {
            "id": "task_easy",
            "name": "Issue Classification",
            "difficulty": "easy",
            "description": "Classify issue as bug, feature, or duplicate. Score 0 or 1."
        },
        {
            "id": "task_medium",
            "name": "Bug Line Identification",
            "difficulty": "medium",
            "description": "Classify + find exact bug line. Partial credit for proximity."
        },
        {
            "id": "task_hard",
            "name": "Full Triage + Fix Suggestion",
            "difficulty": "hard",
            "description": "Classify + bug line + team routing + keyword-checked fix."
        }
    ]

import uvicorn

def main():
    uvicorn.run('server.app:app', host='0.0.0.0', port=7860)

if __name__ == '__main__':
    main()

