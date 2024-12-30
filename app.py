import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model_genai = genai.GenerativeModel("gemini-1.5-flash")

import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import gradio as gr

# Load the whale & dolphin classifier model
model = load_model('./whale_classifier.h5', compile=False)

# Define the labels
labels = {
    0: 'Beluga Whale',
    1: 'Blue Whale',
    2: 'Bottlenose Dolphin',
    3: 'Brydes Whale',
    4: 'Common Dolphin',
    5: 'Cuviers Beaked Whale',
    6: 'Dusky Dolphin',
    7: 'False Killer Whale',
    8: 'Fin Whale',
    9: 'Gray Whale',
    10: 'Humpback Whale',
    11: 'Killer Whale',
    12: 'Long Finned Pilot Whale',
    13: 'Melon Headed Whale',
    14: 'Minke Whale',
    15: 'Pantropic Spotted Dolphin',
    16: 'Pilot Whale',
    17: 'Sei Whale',
    18: 'Short Finned Pilot Whale',
    19: 'Southern Right Whale',
    20: 'Spinner Dolphin',
    21: 'Spotted Dolphin',
    22: 'White Sided Dolphin'
}

# Dictionary containing paths to images for each species
image_paths = {
    'Beluga Whale': 'img/beluga_whale.jpg',
    'Humpback Whale': 'img/humpback_whale.jpg',
    'Killer Whale': 'img/killer_whale.jpg',
    'Bottlenose Dolphin': 'img/bottlenose_dolphin.jpg',
    'Dusky Dolphin': 'img/dusky_dolphin.jpg',
    'Southern Right Whale': 'img/southern_right_whale.jpg',
    'Blue Whale': 'img/blue_whale.jpg',
    'Common Dolphin': 'img/common_dolphin.jpg',
}

# Function to predict whale species
def predict_whale_species(image):
    # Resize the image to match the model's input shape
    img = tf.image.resize(image, (224, 224))
    
    # Convert the image to array and preprocess it
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    answer = model.predict(img)
    y_class = np.argmax(answer)
    predicted_label = labels[y_class]
    image_path = image_paths.get(predicted_label, None)
    
    # Generate structured species information using Gemini API
    prompt = f"Please Summarize the scientific classification, behavior, and conservation status of the {predicted_label} in bullet points."
    
    try:
        response = model_genai.generate_content(prompt)
        species_info = response.text
        return f"## Predicted Species: {predicted_label}\n\n{species_info}", image_path
    except Exception as e:
        return f"Predicted Species: {predicted_label}\n\nError generating species information: {str(e)}", image_path

# Gradio interface description
description = """Upload a picture of the dorsal fin of a whale or dolphin to classify its species."""

with gr.Blocks() as iface:
    gr.Markdown("# Whale & Dolphin Species Classifier")
    gr.Markdown(description)
    
    with gr.Row():
        # First row: Input and Examples
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="pil",  # Ensures consistent image handling
                height=360,  # Set height of input image component
                width=500   # Set width of input image component
            )
            output_text = gr.Textbox(label="Species Classification", visible=False)
            output_image = gr.Image(label="Analyzed Image", visible=False)

        with gr.Column(scale=1):
            examples = gr.Examples(
                examples=[
                    ['examples/1.jpg'],
                    ['examples/2.jpg'],
                    ['examples/3.jpg'],
                    ['examples/4.jpg'],
                    ['examples/5.jpg']
                ],
                inputs=input_image,
                outputs=[output_text, output_image],
                fn=predict_whale_species,
                cache_examples=True
            )
    
    # Submit button between input and output rows
    submit_btn = gr.Button("Submit", size="lg")
    
    with gr.Row():
        # Output row
        with gr.Column(scale=1):
            output_text_main = gr.Markdown(label="Species Classification")
        with gr.Column(scale=1):
            output_image_main = gr.Image(
                label="Analyzed Image",
                height=300,  # Set height of output image component
                width=400   # Set width of output image component
            )
    
    submit_btn.click(
        fn=predict_whale_species,
        inputs=input_image,
        outputs=[output_text_main, output_image_main]
    )

iface.launch(share=True)
