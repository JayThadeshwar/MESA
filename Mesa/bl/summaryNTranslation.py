import googletrans
from transformers import pipeline
from googletrans import Translator

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarizemethod(content):
    translator = Translator()
    summary_text = summarizer(content, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    translated_text = translator.translate(summary_text, src='en', dest='hi')    
    resp = {"summary": summary_text, "translation":translated_text.text}
    return resp