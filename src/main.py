"""Customer Feedback Summarization service"""
import uvicorn
from fastapi import FastAPI
from routes import feedback_summarization
from config import IS_DEVELOPMENT, SERVICE_NAME, API_BASE_URL, PORT
from utils.http_exceptions import add_exception_handlers

BASE_URL = f"/{SERVICE_NAME}/{API_BASE_URL}"

app = FastAPI()


@app.get("/")
def health_check():
  return {
      "success": True,
      "message": "Successfully reached Customer Feedback Summarization service.",
      "data": {}
  }


api = FastAPI(
    title="Customer Feedback Summarization APIs",
    version="latest")

api.include_router(feedback_summarization.router)

app.mount(BASE_URL, api)

add_exception_handlers(app)
add_exception_handlers(api)

if __name__ == "__main__":
  uvicorn.run(
      "main:app",
      host="0.0.0.0",
      port=int(PORT),
      log_level="info",
      reload=IS_DEVELOPMENT)
