"""Inference file to get Customer Feedback Summary."""
from transformers import (TFAutoModelForSeq2SeqLM,
      TFAutoModelForSequenceClassification, AutoTokenizer)
from config import (SUMMARY_MODEL_NAME, SCORING_MODEL_NAME,
      HF_HUB_SUMMARY_MODEL_NAME, HF_HUB_SCORING_MODEL_NAME,
      GENERATION_PARAMS, PREFIX)


summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
scoring_tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL_NAME)

try:
  summary_model = TFAutoModelForSeq2SeqLM.from_pretrained(HF_HUB_SUMMARY_MODEL_NAME)
except:
  summary_model = TFAutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME)

try:
  scoring_model = TFAutoModelForSequenceClassification.from_pretrained(HF_HUB_SCORING_MODEL_NAME)
except:
  scoring_model = TFAutoModelForSequenceClassification.from_pretrained(SCORING_MODEL_NAME)


def get_annotation_score(text, summary):
  """Annotation score for a generated summary.
    Args:
      text: Appended user review on which summary is generated
          Ex: (Twitter: Notion: User: This is a review.)
      summary: Summary for the customer review
          Ex: (User has given a review.)

    Returns:
      annotation_score (float): score according to satisfied guidelines 
  """
  args = (text, summary)
  input_ids = scoring_tokenizer(*args, return_tensors="np")
  annotation_score = round(scoring_model(input_ids).logits.numpy()[0][0], 2)
  return annotation_score


def get_summary_score(request_body):
  """Customer Feedback Summary for a given review.
    Args:
      request_body: Dictionary containing
          customer: Any one of [Notion, figma, zoom]
          type: Any one of [Appstore/Playstore, Twitter, G2]
          feedback: user review for the customer on (type) platform

    Returns:
      Dictionary:
          summary: summarized text
          annotation_score: score according to satisfied guidelines 
  """
  customer = request_body.get("customer", "")
  type = request_body.get("type", "")
  feedback = request_body.get("feedback", "")
  appended_text = ": ".join([type, customer, "User", feedback])
  input_text = PREFIX + appended_text
  input_ids = summary_tokenizer(input_text, return_tensors="tf").input_ids
  outputs = summary_model.generate(input_ids, **GENERATION_PARAMS)
  summarized_text = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)
  annotation_score = get_annotation_score(appended_text, summarized_text)
  return {"summary": summarized_text, "annotation_score": annotation_score}
