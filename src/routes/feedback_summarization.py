"""Customer Feedback Summarization endpoint"""
import traceback
from fastapi import APIRouter
from utils.errors import ValidationError
from utils.http_exceptions import InternalServerError, BadRequest
from utils.logging_handler import Logger
from services.inference import get_summary_score
from schemas.feedback_summarization_schema import RequestModel, ResponseModel
from schemas.error_schema import (InternalServerErrorResponseModel,
                                  ValidationErrorResponseModel)

router = APIRouter(
    tags=["Customer Feedback Summarization"],
    responses={
        500: {
            "model": InternalServerErrorResponseModel
        },
        422: {
            "model": ValidationErrorResponseModel
        }
    })

#pylint: disable=broad-except
@router.post("/summary", response_model=ResponseModel)
def predict(req_body: RequestModel):
  """
  Generates Customer Feedback Summarization

  Args:
    req_body (RequestModel): Required request body for Customer Feedback Summarization

  Raises:
    InternalServerError: 500 Internal Server Error if something fails
    BadRequest: 422 Validation Error if request body is not correct

  Returns:
    [JSON]: Prediction by Customer Feedback Summarization.
    error message if the feedback generation raises an exception
  """
  try:
    summary_score = get_summary_score(req_body.__dict__)
    return {
        "success": True,
        "message": "Successfully generated summary for the given customer feedback",
        "data": summary_score
      }
  except ValidationError as e:
    Logger.error(e)
    Logger.error(traceback.print_exc())
    raise BadRequest(str(e)) from e
  except Exception as e:
    Logger.error(e)
    Logger.error(traceback.print_exc())
    raise InternalServerError(str(e)) from e
