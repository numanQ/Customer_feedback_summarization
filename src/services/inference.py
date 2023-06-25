"""Inference file to get Customer Feedback Summary."""
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from config import MODEL_NAME, CHECKPOINT_PATH, GENERATION_PARAMS, PREFIX
import os

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if os.path.exists(CHECKPOINT_PATH):
  model_name_or_path = CHECKPOINT_PATH
else:
  model_name_or_path = MODEL_NAME

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

def get_summary(request_body):
  """Customer Feedback Summary for a given review."""
  customer = request_body.get("customer", "")
  type = request_body.get("type", "")
  feedback = request_body.get("feedback", "")

  input_text = PREFIX + ": ".join([type, customer, "User", feedback])
  input_ids = tokenizer(input_text, return_tensors="tf").input_ids
  outputs = model.generate(input_ids, **GENERATION_PARAMS)
  summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return summarized_text
