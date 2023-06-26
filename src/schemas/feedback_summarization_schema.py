"""Pydantic models for Customer Feedback Summarization APIs"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from typing_extensions import Literal


class RequestModel(BaseModel):
  """Customer Feedback Summarization request pydantic model"""
  feedback: str = Field(min_length=1)
  customer: Literal["Notion", "figma", "zoom"]
  type: Literal["Appstore/Playstore", "Twitter", "G2"]


  class Config():
    orm_mode = True
    schema_extra = {
      "example": {
        "customer": "Notion",
        "type": "Appstore/Playstore",
        "feedback": "The app doesn't send sound or vibration notifications, which makes this feature almost useless on the iPhone"
      }
    }

class SummaryScore(BaseModel):
  summary: str
  annotation_score: float
class ResponseModel(BaseModel):
  """Customer Feedback Summarization response pydantic model"""
  success: bool
  message: str
  data: Optional[SummaryScore]

  class Config():
    orm_mode = True
    schema_extra = {
      "example": {
        "success": True,
        "message": "Successfully generated summary for the given customer feedback",
        "data": {
            "summary": "None",
            "annotation_score": 0.6
        }
      }
    }
