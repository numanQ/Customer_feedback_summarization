# Customer Feedback Summarization

## Introduction
Summarize customer feedback along with annotation scores

## Prerequisites
Install the required packages

```
pip install git+https://github.com/huggingface/transformers
pip install datasets
pip install tensorflow
pip install evaluate
pip install rouge_score
```
## Train Summarization model
Train a T5 based customer feedback summarization model in tensorflow
```
cd ./src/modeling/
python3 run_summarization.py  \
--model_name_or_path "t5-base" \
--text_column "Appended_text" \
--summary_column "Summary" \
--train_file "./train.csv" \
--validation_file "./val.csv" \
--source_prefix "summarize: " \
--output_dir "./tmp/tst-summarization"  \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--num_train_epochs 40 \
--do_train \
--do_eval \
--num_beams 4
--overwrite_output_dir
```
Here,
 - `text_column`: name of customer review column in csv file
 - `summary_column`: name of summary column in csv file
 - `source_prefix`: Added in the beginning of T5 model describing the task. Here, `summarize: ` 
 - `output_dir`: checkpoint and tensorboard summaries directory
 - `num_beams`: beam width to perform beam search decoding

## Prepare annotation data using external supervision
Using external supervision such as GPT-3.5, annotate few data points in a stratified manner
```
python3 data_prep_annotation_score.py  \
--dataset_filename "./train.csv" \
--annotated_dataset_filename "./annotated_train.csv" \
--annotate_size 200
```

## Train Annotation scoring model of summarized text
Train a DistilBERT based annotation scoring model which generates a score between 0 and 1 based on the number of guidelines which are satisfied
```
python3 run_annotation_scoring.py  \
--train_data "./train.csv" \
--validation_data "./validation.csv" \
--max_length 512 \
--output_dir "./output" \
--text_column "Appended_text" \
--summary_column "Summary" \
--score_column "Annotation_scores" \
--num_epochs 8 \
```
Here,
 - `text_column`: name of customer review column in csv file
 - `summary_column`: name of summary column in csv file
 - `score_column`: name of annotated score column in csv file
 - `output_dir`: checkpoint and tensorboard summaries directory

## Deploy
### Deploy using docker
Build the docker image using the `Dockerfile`:
```
docker build -t feedback_summary .
```
Create a docker container using the docker image
```
docker run -p 7860:7860 feedback_summary
```
This will start a fastapi docker container on
> 0.0.0.0:7860

### Directly deploy the fastapi application
Directly deploy the fastapi service using:
```
cd ./src
uvicorn main:app --host 0.0.0.0 --port 7860
```
This will start a fastapi application on
> 0.0.0.0:7860

# Inference
The following localhost endpoint can be used to fetch the summaries for user feedback and their corresponding annotation scores:
```
[POST]: 0.0.0.0:7860/feedback-summarization/api/v1/summary
``` 
Use the following request body:
```
{
  "customer": "Notion",
  "type": "Appstore/Playstore",
  "feedback": "The application opens, but the pages don't, they load continuously."
}
```

For demo purpose, I have done a complete end to end training of the customer feedback summarization and annotation scoring models and deployed an inference application in huggingface spaces.


You can use the following url for **DEMO**:
[https://numanbot-customer-feedback-summarization.hf.space/feedback-summarization/api/v1/docs#/Customer%20Feedback%20Summarization/predict_summary_post](https://numanbot-customer-feedback-summarization.hf.space/feedback-summarization/api/v1/docs#/Customer%20Feedback%20Summarization/predict_summary_post)
