import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy import load

# Load NLP models and resources
nltk.download('stopwords')
nltk.download('wordnet')
spacy_nlp = load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Load ESG-BERT model and tokenizer
model_name = "nbroad/ESG-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load Sentence Transformer model for text embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Pre-compute embeddings for ESG categories
esg_labels = ["Environment", "Social", "Governance"]
esg_embeddings = embedding_model.encode(esg_labels)

# Helper: Clean and preprocess text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    text = text.strip()  # Trim leading/trailing spaces
    return text

# Helper: Remove stopwords and lemmatize words
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    doc = spacy_nlp(text.lower())  # Convert text to lowercase and process with spaCy
    cleaned_tokens = []

    for token in doc:
        if token.text not in stop_words and token.pos_ != 'PUNCT':
            cleaned_tokens.append(lemmatizer.lemmatize(token.text))

    return " ".join(cleaned_tokens)

# Helper: Map text to ESG category using similarity
def map_to_esg_with_similarity(text):
    text_embedding = embedding_model.encode(text)
    similarities = cosine_similarity([text_embedding], esg_embeddings)
    max_sim_index = np.argmax(similarities)
    return esg_labels[max_sim_index]

# Function to classify text into ESG categories
def classify_esg_text(text):
    text = preprocess_text(clean_text(text))  # Clean and preprocess text
    classification = classifier(text, truncation=True)
    top_label = classification[0]['label']
    esg_category = map_to_esg_with_similarity(top_label)
    return esg_category

# Load CLIP model for image classification
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Helper function to enhance the image for better OCR extraction
def preprocess_image_for_ocr(image):
    # Convert to grayscale
    gray_image = image.convert("L")
    
    # Enhance the image contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2)  # Increase contrast
    
    # Apply thresholding to make text stand out more
    threshold_image = enhanced_image.point(lambda p: p > 128 and 255)  # Binarize the image
    
    # Deskew the image if needed (optional, can add more advanced techniques)
    threshold_image = ImageOps.exif_transpose(threshold_image)
    
    return threshold_image

# Function to classify images
def classify_image(image):
    inputs = clip_processor(text=esg_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()[0]

    max_index = np.argmax(probs)
    return esg_labels[max_index] if probs[max_index] > 0.7 else None

# Extract and classify images from a page
def extract_images_and_classify(page, page_num):
    images_classified = []
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            
            # Preprocess the image to enhance it for OCR
            processed_image = preprocess_image_for_ocr(image)
            
            # Classify image into ESG category
            esg_category = classify_image(processed_image)
            
            # Extract text from the image using OCR
            image_text = pytesseract.image_to_string(processed_image)
            cleaned_image_text = preprocess_text(clean_text(image_text))
            
            if esg_category:
                images_classified.append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "category": esg_category,
                    "image_content": cleaned_image_text if cleaned_image_text else "No text extracted from image"
                })
        except Exception as e:
            print(f"Error extracting image from page {page_num + 1}: {e}")
    
    return images_classified

# OCR text extraction for image-based pages
def extract_text_from_image(page_image_path):
    text = pytesseract.image_to_string(Image.open(page_image_path))
    return preprocess_text(clean_text(text))

# Main extraction and classification pipeline
def extract_and_classify_pdf(pdf_path):
    classified_data = {"E": [], "S": [], "G": []}
    images_classified = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text from page
            text = page.get_text("text").strip()
            if not text:  # If no text, use OCR
                pix = page.get_pixmap()
                image_path = f"temp_page_{page_num}.png"
                pix.save(image_path)
                text = extract_text_from_image(image_path)
                os.remove(image_path)  # Clean up temp image file

            # Classify text
            if text:
                esg_category = classify_esg_text(text)
                classified_data[esg_category[0].upper()].append({"page": page_num + 1, "content": text})

            # Classify images
            page_images = extract_images_and_classify(page, page_num)
            images_classified.extend(page_images)

    # Add classified image results to text categories
    for image_entry in images_classified:
        esg_category = image_entry['category']
        classified_data[esg_category[0].upper()].append({"page": image_entry["page"], "content": image_entry["image_content"]})

    return classified_data

# Save results to JSON
def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {output_file}")

# Example usage
if __name__ == "__main__":
    pdf_path = "./esg_reports/FY2024-NVIDIA-Corporate-Sustainability-Report.pdf"

    # Extract and classify text and images
    classified_data = extract_and_classify_pdf(pdf_path)

    # Save results
    output_file = "data.json"
    save_to_json(classified_data, output_file)