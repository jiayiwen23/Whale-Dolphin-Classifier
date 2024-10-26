import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import gradio as gr
import anthropic
import json
from typing import Tuple, Dict, Optional

# Initialize Anthropic client
client = anthropic.Anthropic()

# Load the model
model = load_model('./whale_classifier.h5', compile=False)

# Define the labels
labels = {
    0: 'Beluga Whale', 1: 'Blue Whale', 2: 'Bottlenose Dolphin', 
    3: 'Brydes Whale', 4: 'Common Dolphin', 5: 'Cuviers Beaked Whale',
    6: 'Dusky Dolphin', 7: 'False Killer Whale', 8: 'Fin Whale',
    9: 'Gray Whale', 10: 'Humpback Whale', 11: 'Killer Whale',
    12: 'Long Finned Pilot Whale', 13: 'Melon Headed Whale',
    14: 'Minke Whale', 15: 'Pantropic Spotted Dolphin',
    16: 'Pilot Whale', 17: 'Sei Whale', 18: 'Short Finned Pilot Whale',
    19: 'Southern Right Whale', 20: 'Spinner Dolphin',
    21: 'Spotted Dolphin', 22: 'White Sided Dolphin'
}

# Dictionary containing paths to images for each species
image_paths = {
    'Beluga Whale': 'beluga_whale.jpg',
    'Humpback Whale': 'humpback_whale.jpg',
    'Killer Whale': 'killer_whale.jpg',
    'Bottlenose Dolphin': 'bottlenose_dolphin.jpg',
    'Dusky Dolphin': 'dusky_dolphin.jpg',
    'Southern Right Whale': 'southern_right_whale.jpg',
    'Blue Whale': 'blue_whale.jpg',
    'Common Dolphin': 'common_dolphin.jpg',
}

def get_species_description(species_name: str) -> Dict:
    """Get detailed species information using Claude function calling"""
    
    functions = [{
        "name": "generate_species_info",
        "description": "Generate detailed information about a marine mammal species",
        "parameters": {
            "type": "object",
            "properties": {
                "scientific_name": {"type": "string"},
                "key_characteristics": {
                    "type": "object",
                    "properties": {
                        "size": {"type": "string"},
                        "weight": {"type": "string"},
                        "lifespan": {"type": "string"},
                        "distinctive_features": {"type": "string"}
                    }
                },
                "behavior": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of behavioral characteristics"
                },
                "habitat": {"type": "string"},
                "diet": {"type": "string"},
                "conservation": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "threats": {"type": "array", "items": {"type": "string"}},
                        "population_trend": {"type": "string"}
                    }
                },
                "interesting_facts": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["scientific_name", "key_characteristics", "behavior", 
                        "habitat", "diet", "conservation", "interesting_facts"]
        }
    }]

    prompt = f"""Generate accurate, detailed information about the {species_name}.
    Include scientific classification, physical characteristics, behavior patterns,
    habitat preferences, diet, conservation status, and interesting facts.
    Focus on unique identifying features and recent scientific findings.
    Format the response according to the provided schema."""

    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            functions=functions
        )
        return json.loads(message.content)
    except Exception as e:
        return {"error": f"Could not generate species description: {str(e)}"}

def format_description(species_info: Dict) -> str:
    """Format the species information into a readable HTML string"""
    if "error" in species_info:
        return f"<div class='error'>{species_info['error']}</div>"
    
    html = f"""
    <div style='font-family: Arial, sans-serif; max-width: 800px; margin: 20px;'>
        <h3 style='color: #2c3e50;'>Scientific Name: {species_info['scientific_name']}</h3>
        
        <h4 style='color: #34495e;'>Key Characteristics:</h4>
        <ul style='list-style-type: none; padding-left: 20px;'>
            <li>📏 Size: {species_info['key_characteristics']['size']}</li>
            <li>⚖️ Weight: {species_info['key_characteristics']['weight']}</li>
            <li>⏳ Lifespan: {species_info['key_characteristics']['lifespan']}</li>
            <li>🔍 Distinctive Features: {species_info['key_characteristics']['distinctive_features']}</li>
        </ul>

        <h4 style='color: #34495e;'>Behavior:</h4>
        <ul>
            {' '.join(f'<li>{behavior}</li>' for behavior in species_info['behavior'])}
        </ul>

        <h4 style='color: #34495e;'>Habitat:</h4>
        <p>{species_info['habitat']}</p>

        <h4 style='color: #34495e;'>Diet:</h4>
        <p>{species_info['diet']}</p>

        <h4 style='color: #34495e;'>Conservation:</h4>
        <ul style='list-style-type: none; padding-left: 20px;'>
            <li>Status: {species_info['conservation']['status']}</li>
            <li>Population Trend: {species_info['conservation']['population_trend']}</li>
            <li>Threats:
                <ul>
                    {' '.join(f'<li>{threat}</li>' for threat in species_info['conservation']['threats'])}
                </ul>
            </li>
        </ul>

        <h4 style='color: #34495e;'>Interesting Facts:</h4>
        <ul>
            {' '.join(f'<li>{fact}</li>' for fact in species_info['interesting_facts'])}
        </ul>
    </div>
    """
    return html

def predict_whale_species(image) -> Tuple[str, Optional[str], str]:
    """Predict species and return classification, example image, and detailed description"""
    # Resize the image to match the model's input shape
    img = tf.image.resize(image, (224, 224))
    
    # Convert the image to array and preprocess it
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    answer = model.predict(img)
    y_class = np.argmax(answer)
    confidence = float(np.max(answer)) * 100
    
    predicted_label = labels[y_class]
    image_path = image_paths.get(predicted_label, None)
    
    # Get detailed species description
    species_info = get_species_description(predicted_label)
    formatted_description = format_description(species_info)
    
    # Add confidence score to the prediction
    prediction_text = f"{predicted_label} (Confidence: {confidence:.1f}%)"
    
    return prediction_text, image_path, formatted_description

# Create the Gradio interface
description = """
<div style='text-align: center;'>
    <h2>🐋 Whale & Dolphin Species Classifier 🐬</h2>
    <p>Upload a picture of a whale or dolphin to classify its species and get detailed information.</p>
    <p>Example images:</p>
    <ul style='list-style-type: none;'>
        <li><a href="https://images.fineartamerica.com/images-medium-large-5/killer-whale-fin-william-ervinscience-photo-library.jpg" target="_blank">Killer Whale</a></li>
        <li><a href="https://www.seakayakadventures.com/sites/seakayakadventures.com/files/images/Beluga-whale-watching.jpg" target="_blank">Beluga Whale</a></li>
        <li><a href="https://bilderreich.de/images/fotos/2016/07/ob/20160703_7237/humpback-whale-dorsal-fin.jpg" target="_blank">Humpback Whale</a></li>
        <li><a href="https://www.researchgate.net/profile/David-Weller-4/publication/305681137/figure/fig2/AS:667697595576326@1536202920527/Dorsal-fin-of-a-coastal-bottlenose-dolphin-off-San-Diego-County-The-numerous-nicks-and.png" target="_blank">Bottlenose Dolphin</a></li>
    </ul>
</div>
"""

iface = gr.Interface(
    fn=predict_whale_species,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Text(label="Predicted Species"),
        gr.Image(label="Example Image"),
        gr.HTML(label="Species Information")
    ],
    title="Whale & Dolphin Species Classifier",
    description=description,
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 900px; margin: auto;}"
)

if __name__ == "__main__":
    iface.launch(share=True)
