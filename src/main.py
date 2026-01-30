import uvicorn
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Parking LLM Microservice")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
