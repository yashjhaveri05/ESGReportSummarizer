# Import necessary libraries
import os
from bert_score import score
from rouge import Rouge

# Read the summaries from the text files
with open('sample_single.txt', 'r') as file:
    summary_single_agent = file.read()
with open('sample_multi.txt', 'r') as file:
    summary_multi_agent = file.read()

# BERTScore
candidate_summaries = [summary_single_agent]
reference_summaries = [summary_multi_agent]
P, R, F1 = score(candidate_summaries, reference_summaries, lang='en', verbose=True)
print(f"\nBERTScore:")
print(f"Precision: {P.mean().item():.4f}")
print(f"Recall: {R.mean().item():.4f}")
print(f"F1 Score: {F1.mean().item():.4f}")

# ROUGE Score
rouge = Rouge()
scores = rouge.get_scores(summary_single_agent, summary_multi_agent)
print(f"\nROUGE Scores:")
print(scores)

# Length
len_single = len(summary_single_agent.split())
len_multi = len(summary_multi_agent.split())
print(f"\nSummary Lengths (number of words):")
print(f"Single Agent Length: {len_single}")
print(f"Multi-Agent Length: {len_multi}")