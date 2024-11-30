# Import necessary libraries
import os
from bert_score import score
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from textblob import TextBlob
from gensim import corpora, models
from transformers import pipeline

# Read the summaries from the text files
with open('single.txt', 'r') as file:
    summary_single_agent = file.read()

with open('multi.txt', 'r') as file:
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

# Embedding Cosine Similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([summary_single_agent, summary_multi_agent])
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"\nEmbedding Cosine Similarity: {similarity:.4f}")

# Readability Metrics
readability_single = textstat.flesch_reading_ease(summary_single_agent)
readability_multi = textstat.flesch_reading_ease(summary_multi_agent)
print(f"\nReadability Scores (Flesch Reading Ease):")
print(f"Single Agent Readability: {readability_single:.2f}")
print(f"Multi-Agent Readability: {readability_multi:.2f}")

# Sentiment Analysis
sentiment_single = TextBlob(summary_single_agent).sentiment.polarity
sentiment_multi = TextBlob(summary_multi_agent).sentiment.polarity
print(f"\nSentiment Polarity Scores:")
print(f"Single Agent Sentiment Polarity: {sentiment_single:.2f}")
print(f"Multi-Agent Sentiment Polarity: {sentiment_multi:.2f}")

# Length
len_single = len(summary_single_agent.split())
len_multi = len(summary_multi_agent.split())
print(f"\nSummary Lengths (number of words):")
print(f"Single Agent Length: {len_single}")
print(f"Multi-Agent Length: {len_multi}")

# Topic Modeling
texts = [summary_single_agent.lower().split(), summary_multi_agent.lower().split()]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Build the LDA model
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42)
print(f"\nTopics in Single Agent Summary:")
topics_single = lda_model.show_topics(formatted=False, num_words=5)
print(topics_single)

# Since we have only two documents, the same model is used for both
print(f"\nTopics in Multi-Agent Summary:")
topics_multi = lda_model.show_topics(formatted=False, num_words=5)
print(topics_multi)

# NLI-based Evaluation
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["paraphrase", "not paraphrase"]
print("\nNLI-based Evaluation:")
result_single_hypo = nli_model(summary_single_agent, candidate_labels=labels, hypothesis_template="This text is {}.")
result_multi_hypo = nli_model(summary_multi_agent, candidate_labels=labels, hypothesis_template="This text is {}.")
print(f"Single Agent Summary NLI Result:")
print(result_single_hypo)
print(f"\nMulti-Agent Summary NLI Result:")
print(result_multi_hypo)

# Comparing if one summary entails the other
nli_result = nli_model(summary_single_agent, candidate_labels=["entailment", "contradiction", "neutral"], hypothesis_template="{}")
print(f"\nNLI Comparison Result between Summaries:")
print(nli_result)