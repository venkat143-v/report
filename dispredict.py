import pandas as pd
import cv2
import fitz
import numpy as np
import tensorflow as tf
from tensorflow import keras
import spacy
import os
import pytesseract

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()
        text.append(page_text)
    return text


# Function to preprocess text using spaCy
def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    processed_text = " ".join(tokens)
    return processed_text

def disease_predict(file_path):
    def extract_text(file_path):
        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension=='.pdf':
            text = []
            pdf_document = fitz.open(file_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                text.append(page_text)
            return text
        else:
            image = cv2.imread(file_path)
            text_from_image = pytesseract.image_to_string(image)
            return [text_from_image]

    file_extension = os.path.splitext(file_path)[-1].lower()

    combined_text = extract_text(file_path)
    max_sequence_length = 100
    vocabulary_size = 10000
    embedding_dim = 32
    
    # Move the tokenizer creation outside of the loop
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size, output_mode="int")
    tokenizer.adapt(combined_text)  # Adapt to the combined text
    ans=[]
    for page_num, processed_text in enumerate(combined_text):
        sequences = tokenizer(processed_text)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences([sequences], maxlen=max_sequence_length)

        model = keras.Sequential([
            keras.layers.Input(shape=(max_sequence_length,)),
            keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim),
            keras.layers.LSTM(128),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        predicted_disease_prob = model(padded_sequences)  # Call the model directly on tensors
        predicted_disease_label = 1 if predicted_disease_prob >= 0.5 else 0
        ans.append(predicted_disease_label)
    return ans
