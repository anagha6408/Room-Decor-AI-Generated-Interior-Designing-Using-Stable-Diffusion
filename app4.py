import os
from flask import Flask, request, send_file, url_for, jsonify
from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image
import io
import random

app = Flask(__name__)

# Load your fine-tuned model
model_id = "my-ctrlnet"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# Define styles and associated prompts
styles_and_prompts = {
    "modern": [
        "A sleek, modern living room with minimalistic decor",
        "A spacious modern interior with natural lighting",
        "Contemporary room with neutral colors and modern furniture",
        "Modern living space with elegant, simple furnishings"
    ],
    "bohemian": [
        "A colorful bohemian room with eclectic decor",
        "Boho style room with vibrant colors and patterns",
        "Relaxed, cozy space with bohemian decor",
        "Bohemian interior with plants and textiles"
    ],
    "industrial": [
        "Industrial-style room with exposed brick and metal fixtures",
        "A rugged, industrial living space with a minimalist vibe",
        "Room with industrial decor, concrete walls, and metal furniture",
        "Loft-style industrial room with modern lighting"
    ],
    "minimalist": [
        "A minimalist room with simple, clean lines and neutral colors",
        "Minimalist decor with few furnishings and open space",
        "Calm, uncluttered room with natural textures and minimal decor",
        "Minimalist interior focusing on functionality and simplicity"
    ]
}

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/generate', methods=['POST'])
def generate_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Validate file selection
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
    file.seek(0, os.SEEK_END)  # Move to end of file
    file_size = file.tell()
    file.seek(0)  # Reset pointer to the start of the file
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"File exceeds the maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB"}), 400
    
    # Validate style
    style = request.form.get('style')
    if not style:
        return jsonify({"error": "No style provided"}), 400
    if style not in styles_and_prompts:
        return jsonify({"error": "Invalid style provided"}), 400
    
    # Randomly select a prompt from the style's prompt list
    prompt = random.choice(styles_and_prompts[style])

    # Open the uploaded image
    image = np.array(Image.open(file.stream))
    
    # Canny edge detection
    low_threshold = 150
    high_threshold = 250
    image = cv2.Canny(image, low_threshold, high_threshold)
    
    # Convert Canny output to RGB format for the model
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate the output image with the model
    output = pipeline(prompt=prompt, image=canny_image, num_inference_steps=16).images[0]
    
    # Determine the next file name in sequence
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of files in the output directory and find the highest numbered file
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("output_image_") and f.endswith(".png")]
    existing_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files if f.split("_")[2].split(".")[0].isdigit()]
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Save the output image locally
    output_path = os.path.join(output_dir, f"output_image_{next_number}.png")
    output.save(output_path)
    
    # Get all image filenames in the output directory
     # Get the URL of the latest generated image only
    image_url = url_for('serve_image', filename=f"output_image_{next_number}.png", _external=True)
    
    # Return only the URL of the latest generated image in the response
    return jsonify({"image_url": image_url}), 200

@app.route('/images/<filename>')
def serve_image(filename):
    # Serve the requested image file from the output_images directory
    output_dir = 'output_images'
    file_path = os.path.join(output_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
