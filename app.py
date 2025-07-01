import os  # Add this import
from flask import Flask, request, send_file
from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your fine-tuned model
model_id = "my-ctrlnet"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/generate', methods=['POST'])
def generate_image():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

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
     # Check if a prompt was provided
    prompt = request.form.get('prompt')
    if not prompt:
        return "No prompt provided", 400
    
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
    
    # Save the output image to a byte stream for returning
    img_byte_arr = io.BytesIO()
    output.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Return the image as a file in the response
    return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='output_image.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
