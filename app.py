from flask import Flask, render_template, request, redirect, url_for
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load Azure keys and endpoints from environment variables

COMPUTER_VISION_KEY = "65G4nMtns49pZtNDRXByTExMOU0IlkQHrgyCScrOgaKjcheJDCo3JQQJ99BAACYeBjFXJ3w3AAAFACOGuwO0"
COMPUTER_VISION_ENDPOINT = "https://myvisionaiservice001.cognitiveservices.azure.com/"
TRANSLATOR_KEY = "295ABoWRGSVjlBZ9sLIri8kQ5gCUt0D2j1nNsOS8TyHLu2qXpGlrJQQJ99BAACYeBjFXJ3w3AAAbACOGqm7i"
TRANSLATOR_ENDPOINT = "https://myaitranslatorservice001.cognitiveservices.azure.com/"

# Initialize Azure Vision Client
vision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_KEY))

# Initialize Azure Translator Client
translator_client = TextTranslationClient(
    endpoint=TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(TRANSLATOR_KEY)
)

# Index route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and OCR processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(request.url)
    
    # Create a unique filename to avoid overwriting
    image_filename = str(uuid.uuid4()) + "_" + image_file.filename
    image_path = os.path.join('static/images', image_filename)
    image_file.save(image_path)
    
    # Perform OCR on the uploaded image
    detected_text, detected_language, ocr_error_message = perform_ocr(image_path)
    
    # If OCR failed, show an error message
    if ocr_error_message:
        return render_template('index.html', 
                               image_url=image_path, 
                               detected_text=detected_text, 
                               detected_language=detected_language, 
                               translated_text="",
                               ocr_error_message=ocr_error_message)
    
    # Translate the detected text to English
    translated_text = translate_text(detected_text, detected_language)
    
    # Return the results in the same page with the uploaded image
    return render_template('index.html', 
                           image_url=image_path, 
                           detected_text=detected_text, 
                           detected_language=detected_language, 
                           translated_text=translated_text,
                           ocr_error_message="")

# Function to perform OCR using Azure Cognitive Services Computer Vision API
def perform_ocr(image_path):
    try:
        with open(image_path, 'rb') as image:
            # Use the correct method: recognize_printed_text_in_stream
            result = vision_client.recognize_printed_text_in_stream(image, language='unk')  # Use 'unk' for auto language detection

        detected_text = ""
        detected_language = "Unknown"
        ocr_error_message = None
        
        # If no text is detected, set an error message
        if not result.regions:
            ocr_error_message = "No text detected in the image."
            return detected_text, detected_language, ocr_error_message
        
        # Iterate through the OCR result to get detected text
        for region in result.regions:
            for line in region.lines:
                detected_text += " ".join([word.text for word in line.words]) + "\n"
        
        # Azure Computer Vision API gives language info in 'result.language'
        if result.language:
            detected_language = result.language
        
        return detected_text.strip(), detected_language, ocr_error_message
    
    except Exception as e:
        print(f"Error in OCR: {e}")
        return "Error during OCR processing.", "Unknown", "Failed to process the image. Please try again."

# Function to translate detected text using Azure Translator API
def translate_text(text, detected_language):
    try:
        # Prepare the request body as a list of dictionaries with 'text' keys
        request_body = [{"text": text}]
        
        # If the detected language is one of the Indian regional languages, translate to English
        if detected_language not in ['en', 'unk']:  # 'unk' is for unknown language
            # Translate text to English
            response = translator_client.translate(
                body=request_body,
                from_language=detected_language,
                to_language=["en"]  # Target language is English
            )
        else:
            # If detected language is English or unknown, no translation needed
            return text.strip()
        
        translated_text = ""
        for translation in response:
            translated_text += translation.translations[0].text + "\n"
        
        return translated_text.strip()
    
    except Exception as e:
        print(f"Error in Translation: {e}")
        return "Error during translation."

if __name__ == "__main__":
    app.run(debug=True)
