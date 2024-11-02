# backend/app/routes.py

from flask import request, jsonify
from flask import current_app as app
from .ai_models import HuggingFaceModel, OpenAIModel
import os

# Initialize AI Models using environment variables
huggingface_summarizer = HuggingFaceModel(task='summarization', model_name='facebook/bart-large-cnn')
huggingface_sentiment = HuggingFaceModel(task='sentiment-analysis', model_name='distilbert-base-uncased-finetuned-sst-2-english')
openai_paraphraser = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), task='paraphrasing')
openai_question_generator = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), task='question_generation')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    analyses = data.get('analyses', [])

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    results = {}

    # Perform selected analyses
    if 'summary' in analyses:
        summary = huggingface_summarizer.generate_summary(text)
        results['summary'] = summary

    if 'key_points' in analyses:
        key_points = extract_key_points(text)
        results['key_points'] = key_points

    if 'sentiment_analysis' in analyses:
        sentiment = huggingface_sentiment.analyze_sentiment(text)
        results['sentiment'] = sentiment

    if 'paraphrasing' in analyses:
        paraphrased_text = openai_paraphraser.paraphrase(text)
        results['paraphrased_text'] = paraphrased_text

    if 'question_extraction' in analyses:
        questions = openai_question_generator.generate_questions(text)
        results['generated_questions'] = questions

    return jsonify(results), 200

@app.route('/answer_question', methods=['POST'])
def answer_question_route():
    data = request.get_json()
    text = data.get('text', '')
    question = data.get('question', '')
    
    if not text or not question:
        return jsonify({'error': 'Text and question must be provided.'}), 400
    
    answer = openai_question_generator.answer_question(question, text)
    return jsonify({'answer': answer}), 200

def extract_key_points(text):
    # Simple extraction using summarization; can be enhanced with more sophisticated NLP
    summarizer = HuggingFaceModel(task='summarization', model_name='facebook/bart-large-cnn')
    summary = summarizer.generate_summary(text, max_length=300)
    # Split summary into key points based on sentence boundaries
    key_points = summary.split('. ')
    key_points = [point.strip() + '.' for point in key_points if point]
    return key_points
