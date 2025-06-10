import os
import pandas as pd
import numpy as np
import re
import io
import tempfile
import traceback # Added for detailed error logging

# --- Flask Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Pinecone Imports ---
from pinecone import Pinecone, Index

# --- PyTorch Imports for CNN Model ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms # For image preprocessing
from PIL import Image # For loading images

# --- Sentence Transformers for Embeddings ---
from sentence_transformers import SentenceTransformer

# --- Import your images.py file for OCR functionality ---
import images

# REMOVED: import tensorflow as tf

from dotenv import load_dotenv
load_dotenv()

print(f"PyTorch Version: {torch.__version__}")
# REMOVED: print(f"TensorFlow Version: {tf.__version__}")
# Check for GPU with PyTorch
if torch.cuda.is_available():
    print(f"CUDA is available! PyTorch using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. PyTorch using CPU.")
# REMOVED: print(f"Num GPUs Available (TensorFlow): {len(tf.config.experimental.list_physical_devices('GPU'))}")


# --- Global Embedding Model Variable ---
model_for_embeddings = None

# Function to get embedding for a text
def get_embedding(text):
    """
    Generates an embedding vector for the given text using your chosen model.
    """
    if not text:
        return None
    try:
        if model_for_embeddings is not None:
            return model_for_embeddings.encode(text).tolist()
        else:
            print("ERROR: Embedding model 'all-MiniLM-L6-v2' not loaded. Cannot generate real embeddings.")
            # Return a dummy embedding if the model isn't loaded, for testing purposes
            dummy_embedding_dimension = 384
            return np.random.rand(dummy_embedding_dimension).tolist()

    except Exception as e:
        print(f"Error generating embedding for text: '{text}' - {e}")
        return None

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "product-recommendation-index"

# --- CNN Model Configuration ---
CNN_MODEL_SAVE_PATH = 'cnn_product_classifier.pth' 
BASE_IMAGE_DIR = 'product_images'          
IMG_HEIGHT = 128                          
IMG_WIDTH = 128                           
IMG_SIZE = IMG_HEIGHT                     



product_data_map = {}
pinecone_client = None
pinecone_index = None
cnn_model_pytorch = None
cnn_class_names = []


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)



cnn_image_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])


def clean_stock_code_for_lookup(stock_code):
    if pd.isna(stock_code):
        return None
    stock_code = str(stock_code)
    cleaned = re.sub(r'[^\w]', '', stock_code)
    return cleaned

# --- Core Recommendation Logic Helper Function ---
def get_product_recommendations_logic(user_query):
    print(f"DEBUG: Recommendation Logic: Processing query: '{user_query}'")

    if not user_query or not isinstance(user_query, str):
        return {
            "status": "error",
            "message": "Invalid query: 'query' field is required and must be a string."
        }, 400

    if len(user_query) > 500:
        return {
            "status": "error",
            "message": "Query too long. Please keep your query concise (max 500 characters)."
        }, 400
    
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return {
            "status": "error",
            "message": "Failed to generate embedding for the query. Please ensure your embedding model is configured correctly."
        }, 500

    recommended_products_list = []
    if pinecone_index:
        try:
            query_results = pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            print(f"DEBUG: Pinecone Query Results: {query_results}")

            for match in query_results.matches:
                stock_code = match.metadata.get('StockCode', 'N/A')
                description = match.metadata.get('Description', 'Description not found in Pinecone metadata')
                unit_price = match.metadata.get('UnitPrice', 'N/A') # Use 'UnitPrice' or 'Unit_Price' as per your data
                
                recommended_products_list.append({
                    "stock_code": stock_code,
                    "description": description,
                    "unit_price": unit_price,
                    "similarity_score": match.score
                })

            natural_language_response = f"Based on your query: '{user_query}', here are some product recommendations:"

        except Exception as e:
            print(f"DEBUG: Error during Pinecone query or result processing: {e}")
            return {
                "status": "error",
                "message": "Error retrieving recommendations from the database."
            }, 500
    else:
        natural_language_response = "Recommendation service is currently unavailable. Pinecone index not initialized."
        print(f"DEBUG: Pinecone index not initialized in recommendation logic.")
        return {
            "status": "error",
            "message": natural_language_response
        }, 503

    return {
        "status": "success",
        "message": natural_language_response,
        "recommended_products": recommended_products_list
    }, 200

# --- OCR FUNCTION ---
def perform_ocr_on_image(image_path):
    try:
        extracted_text = images.extract_text_from_image(image_path)
        return extracted_text
    except Exception as e:
        print(f"OCR Error calling images.py: {e}")
        return None

# --- Flask Routes ---

@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    data = request.get_json()
    user_query = data.get('query', '')
    
    print(f"DEBUG: Endpoint /recommend_products: Received query: '{user_query}'")
    
    response_data, status_code = get_product_recommendations_logic(user_query)
    return jsonify(response_data), status_code

@app.route('/ocr_recommend_products', methods=['POST'])
def ocr_recommend_products():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided. Please upload an image with key 'image'."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"status": "error", "message": "No selected image file."}), 400

    allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    if not image_file.filename.lower().endswith(allowed_extensions):
        return jsonify({"status": "error", "message": f"Unsupported file type. Please upload an image ({', '.join(allowed_extensions)})."}), 400

    extracted_text = None
    temp_image_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_img:
            image_file.save(temp_img.name)
            temp_image_path = temp_img.name
        
        print(f"DEBUG: OCR Endpoint: Image saved temporarily at {temp_image_path}")
        extracted_text = perform_ocr_on_image(temp_image_path)
        
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
            print(f"DEBUG: OCR Endpoint: Temporary image file {temp_image_path} deleted.")

        if extracted_text is None or not extracted_text.strip():
            return jsonify({
                "status": "error",
                "message": "Could not extract clear text from the image. Please ensure the text is readable."
            }), 400

        print(f"DEBUG: OCR Endpoint: Extracted text: '{extracted_text}'")
        response_data, status_code = get_product_recommendations_logic(extracted_text)
        response_data['extracted_text'] = extracted_text
        
        return jsonify(response_data), status_code

    except Exception as e:
        print(f"Error in OCR-based query processing: {e}")
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        return jsonify({"status": "error", "message": f"Internal server error during OCR processing: {e}"}), 500

@app.route('/image_detect_products', methods=['POST'])
def image_detect_products():
    """
    Endpoint for image-based product detection using CNN and then vector database matching.
    Input: Product image (multipart/form-data) with key 'image'.
    Output: Product description and matching products, plus the CNN-identified class name.
    """
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided. Please upload an image with key 'image'."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"status": "error", "message": "No selected image file."}), 400

    allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    if not image_file.filename.lower().endswith(allowed_extensions):
        return jsonify({"status": "error", "message": f"Unsupported file type. Please upload an image ({', '.join(allowed_extensions)})."}), 400

    temp_image_path = None
    predicted_class_name = "Could not identify product from image."

    try:
        if cnn_model_pytorch is None:
            print("ERROR: cnn_model_pytorch is None. Model failed to load during startup.")
            return jsonify({
                "status": "error",
                "message": "PyTorch CNN model not loaded. Please check server logs for details."
            }), 500

        # Save image temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_img:
            image_file.save(temp_img.name)
            temp_image_path = temp_img.name
        
        print(f"DEBUG: Image Detection Endpoint: Image saved temporarily at {temp_image_path}")


        img = Image.open(temp_image_path).convert('RGB') 
        input_tensor = cnn_image_transform(img)          
        input_batch = input_tensor.unsqueeze(0)          


        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            cnn_model_pytorch.to('cuda')

        with torch.no_grad(): 
            output = cnn_model_pytorch(input_batch)
            probabilities = F.softmax(output, dim=1) 
        
        confidence, predicted_class_index_tensor = torch.max(probabilities, 1)
        predicted_class_index = predicted_class_index_tensor.item()
        confidence_val = confidence.item()

        if torch.cuda.is_available():
            cnn_model_pytorch.to('cpu')

        if cnn_class_names and predicted_class_index < len(cnn_class_names):
            predicted_class_name = cnn_class_names[predicted_class_index]
            print(f"DEBUG: CNN Predicted: '{predicted_class_name}' with confidence: {confidence_val:.2f}")
        else:
            print(f"WARNING: Predicted class index {predicted_class_index} out of bounds for cnn_class_names (length {len(cnn_class_names)}).")
            predicted_class_name = "Unknown Product (CNN prediction error)"

        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
            print(f"DEBUG: Image Detection Endpoint: Temporary image file {temp_image_path} deleted.")
        response_data, status_code = get_product_recommendations_logic(predicted_class_name)
        response_data['cnn_identified_class'] = predicted_class_name
        response_data['cnn_confidence'] = confidence_val
        
        return jsonify(response_data), status_code

    except Exception as e:
        print(f"ERROR in image_detect_products endpoint: {e}")
        traceback.print_exc()
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        return jsonify({"status": "error", "message": f"Internal server error during image detection: {e}"}), 500

if __name__ == '__main__':
    
    if not product_data_map:
        try:
            df_cleaned = pd.read_csv('cleaned_dataset.csv')
            df_cleaned['StockCode'] = df_cleaned['StockCode'].astype(str)
            for index, row in df_cleaned.iterrows():
                cleaned_sc = row['StockCode']
                description = row['Description']
                if pd.notna(description) and cleaned_sc not in product_data_map:
                    product_data_map[cleaned_sc] = description
            print("Product data loaded for description lookup from cleaned_dataset.csv.")
        except FileNotFoundError:
            print("WARNING: cleaned_dataset.csv not found. Product descriptions will not be available via map.")
        except Exception as e:
            print(f"ERROR loading product data from cleaned_dataset.csv: {e}")
            traceback.print_exc() 
    if pinecone_index is None:
        try:
            if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
                pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
                pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
                print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                print("WARNING: Pinecone API key or environment not set. Please set environment variables 'PINECONE_API_KEY' and 'PINECONE_ENVIRONMENT'.")
        except Exception as e:
            print(f"ERROR connecting to Pinecone: {e}")
            traceback.print_exc() 
            pinecone_client = None
            pinecone_index = None


    if model_for_embeddings is None:
        try:
            print(f"Loading embedding model: 'all-MiniLM-L6-v2'...")
            model_for_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
        except Exception as e:
            print(f"ERROR loading embedding model 'all-MiniLM-L6-v2': {e}")
            traceback.print_exc() 
            model_for_embeddings = None

  
    if cnn_model_pytorch is None:
        try:
           
            if os.path.exists(BASE_IMAGE_DIR):
                print(f"DEBUG: Attempting to infer CNN class names from '{BASE_IMAGE_DIR}'...")
               
                all_items = os.listdir(BASE_IMAGE_DIR)
              
                inferred_class_names = sorted([d for d in all_items if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))])
                
                cnn_class_names.clear()
                cnn_class_names.extend(inferred_class_names)
                print(f"DEBUG: Inferred {len(cnn_class_names)} CNN class names: {cnn_class_names}")
            else:
                print(f"WARNING: Base image directory '{BASE_IMAGE_DIR}' not found. Cannot infer CNN class names. Ensure it exists and contains subfolders of images.")

          
            if cnn_class_names: 
                model_architecture = SimpleCNN(num_classes=len(cnn_class_names))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_architecture.to(device)

                if os.path.exists(CNN_MODEL_SAVE_PATH):
                    print(f"DEBUG: Attempting to load CNN model state_dict from '{CNN_MODEL_SAVE_PATH}' on {device}...")
                    model_architecture.load_state_dict(torch.load(CNN_MODEL_SAVE_PATH, map_location=device))
                    model_architecture.eval() 
                    cnn_model_pytorch = model_architecture
                    print(f"DEBUG: PyTorch CNN Model '{CNN_MODEL_SAVE_PATH}' loaded successfully.")
                else:
                    print(f"WARNING: PyTorch CNN Model not found at '{CNN_MODEL_SAVE_PATH}'. Image detection endpoint will not work. Please ensure the model file is in the correct directory.")
                    cnn_model_pytorch = None
            else:
                print("WARNING: CNN class names not inferred or empty. Image detection endpoint will not work.")
                cnn_model_pytorch = None

        except Exception as e:
            print(f"ERROR: Failed to load PyTorch CNN model or infer class names: {e}")
            traceback.print_exc() 
            cnn_model_pytorch = None

    print("\nFlask app starting.")
    print("Access the text-based endpoint at http://127.0.0.1:5000/recommend_products")
    print("Access the OCR-based endpoint at http://127.0.0.1:5000/ocr_recommend_products")
    print("Access the Image-based endpoint at http://127.0.0.1:5000/image_detect_products")
    app.run(debug=True, port=5000)