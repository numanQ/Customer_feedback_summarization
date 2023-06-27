import pandas as pd
import numpy as np
import openai
import datetime
import time
import argparse
from fractions import Fraction

openai.api_key = """<OPENAI_API_KEY>"""

annotation_context = """We provide you, with the annotation guidelines that our annotators have used to write summaries for records.
The Annotation Guidelines should help formulate an idea of what the ideal summary for a record looks like. Make sure you go through the entire contents of the Annotation Guidelines properly as it dictates what is expected in a summary across a variety of Record Types.

1. Read the entire content carefully: Before deciding whether to create a summary, ensure that you have read and understood the whole review, survey, support conversation, app store/play store review, NPS, or user recording to get a clear grasp of the user's concerns, expectations, and feedback.
2. Evaluate the usefulness of the content: Determine if the content provides meaningful information or feedback that would be valuable to the product team. If the content lacks substantial information or is just a conversation with a happy resolution and noactionable insights, the summary should be "None." If the content has some information but is not detailed enough to be actionable, try to summarise the general issue or concern without assuming additional details.
3. Identify the "What": If the content is deemed valuable, determine the main subject of the content, which is usually the product, service, or specific feature being discussed. Summarise the user's experience or opinion about it in a few words.
4. Address the "Why": Explain briefly why the user has the opinion they expressed in the content. This may include any specific reasons, challenges, or issues they faced while using the product or service. If the "Why" is not explicitly mentioned but can bereasonably inferred, include it in the summary.
5. Mention the "How": If the content provides any solutions or suggestions, include them in your summary. This may involve the user's recommendations, steps they took to resolve an issue, or alternative methods they tried.
6. Keep it concise: The summary should be short, yet informative. Aim to capture the essence of the content in one or two sentences, without using unnecessary words or repeating information.
7. Use clear and simple language: Write the summary using clear and straightforward language, avoiding jargon or complex terms. Make sure it is easily understandable by a wide audience.
8. Maintain objectivity: Ensure that your summary is unbiased and does not include your personal opinions.
9. Adapt to the content type: While the general guidelines remain the same, be prepared to adapt your summary to the specific type of content you are summarising. For example, support conversation tickets may require a focus on the resolution and steps taken, while app store/play store reviews may require highlighting the user's overall satisfaction and any standout features.
10. Highlight key insights: In case of surveys or NPS, emphasize the key insights or trends that emerge from the user's responses. For user recordings, focus on the most important takeaways from the user's interaction with the product or service.
11. Ensure accuracy: Make sure your summary accurately represents the user's experience and opinions, without distorting or exaggerating their feedback.
12. Focus on the user's issue: When summarising support conversations, prfioritize the user's problem or concern, rather than the agent's actions, to ensure the summary is useful for understanding the user's issue.
13. Maintain a consistent narrative: Use a consistent narrative style throughout the summary, regardless of the feedback source. Start with the user's issue or opinion, followed by the reasons or challenges they faced, and end with any solutions or suggestions provided. Avoid using questions or statements that do not convey a clear understanding of the user's issue or opinion.
14. Capture implied information: If the user's feedback implies a reason for their opinion or issue, try to include it in the summary, even if it is not explicitly stated. This will help provide a more comprehensive understanding of the user's experience.
15. While summarising a Twitter review fetch the last sent message , use the context of the message to summarise the last message, don't mention any personal information like username , twitter handle in the summary.
16. Understand the questions asked by Agent and Summarise the User Response Accordingly ,use the Agent Question to draw context for the answers.

Using the above guidelines, return a final score using the following rules:
1. Maximum_total_score = the total number of annotation guidelines
2. Text_score = the number of annotation guidelines satisfied
3. Final_score = Text_Score/Maximum_total_score
"""
system_message = {"role": "system", "content": "You have to only reply as\nScore:<Final_score>"}

def prepare_dataset(dataset_filename):
  """Prepare dataset for annotation labeling"""
  # raw_dataset_filename = "./train_raw/train_raw.csv"
  # Load the raw data on which to generate the annotation scores
  raw_dataset = pd.read_csv(dataset_filename)
  # Stratify the data into 10 bins based on semantic similarity score of user review and summary text
  similarity_score_category = pd.cut(
    raw_dataset.Similarity_scores, bins=list(np.arange(0, 1.1, 0.1)), labels = list(range(10)))
  # Add the bins as a separate categorical variable named "Similarity_score_category"
  raw_dataset.insert(5, "Similarity_score_category", similarity_score_category)
  raw_dataset["Annotation_scores"] = np.nan
  return raw_dataset


def get_stratified_dataframe_indexes(df, category=None, total_data=200):
  """Get indexes of data belonging to the category such that it has the
    same proportion in <total_data> as the full data
    Ex: category is 1 which has 71 rows out of 4000, then we need to return
        randomly indexes of 71/4000*200 indexes for that category
  """
  if category==None:
    stratified_df = df[df["Summary"]=="None"]
  else:
    stratified_df = df[df["Similarity_score_category"]==category]
  category_sample_size = int(len(stratified_df)/len(df)*total_data)
  idxs = stratified_df.sample(n=category_sample_size).index.to_list()
  return idxs

def gpt_supervision_score(type, text, summary):
  """Get annotation score for a user feedback text and summary.
     Here we have used GPT-3.5-turbo
     Args:
        type: any one of ["Appstore/Playstore", "Twitter", "G2"]
        text: user review for the customer
        summary: summarized user review

     Returns:
        reply: GPT-3.5 generated score based on the annotation guidelines
        time_taken: time taken to generate (in secs)
  """
  messages = [system_message]
  message = annotation_context + "\n\n" + \
    f"Type:: {type}\nText:: {text}\nSummary:: {summary}\n"
  messages.append({"role":"user", "content": message})
  start_time = datetime.datetime.now().replace(microsecond=0)
  chat = openai.ChatCompletion.create(
           model="gpt-3.5-turbo", messages=messages)
  end_time = datetime.datetime.now().replace(microsecond=0)
  time_taken = end_time-start_time
  print("Time taken = ", (time_taken))
  reply = chat.choices[0].message.content
  return reply, time_taken.seconds


def impute_annotation_score(df, idx):
  """Impute the annotation score generated by external supervision
     in `Annotation_scores` corresponding to `idx` row"""
  annotation_score, time_taken = gpt_supervision_score(
    df.loc[idx, "Type"], df.loc[idx, "Text"], df.loc[idx, "Summary"])
  print(annotation_score)
  try:
    score_str = annotation_score.split("Score:")[1].strip().split()[0]
    score = float(sum(Fraction(s) for s in score_str.split()))
    df.loc[idx, "Annotation_scores"] = score
  except:
    df.loc[idx, "Annotation_scores"] = annotation_score
  return time_taken


def main():
  # Parse arguments from cmd
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_filename", help = "name of the dataset (csv) required to annotate scores")
  parser.add_argument("--annotate_size", help = "Size of the data to annotate", default = 200, type = int)
  parser.add_argument("--annotated_dataset_filename", help = "name of the annotated dataset (csv)")
  args = parser.parse_args()

  raw_dataset = prepare_dataset(args.dataset_filename)

  # Get all indexes of rows to annotate
  indexes = []
  for category in raw_dataset.Similarity_score_category.unique():
    if np.isnan(category):
      idxs = get_stratified_dataframe_indexes(raw_dataset, None, args.annotate_size)
    else:
      idxs = get_stratified_dataframe_indexes(raw_dataset, category, args.annotate_size)
    indexes.extend(idxs)

  # Get annotation scores using external supervision on the selected indexes
  for i, idx in enumerate(indexes):
    current_time_taken = impute_annotation_score(raw_dataset, idx)
    print("Completed: ", i+1)
    print()
  raw_dataset.to_csv(args.annotated_dataset_filename, index=False)

if __name__ == "__main__":
  main()
