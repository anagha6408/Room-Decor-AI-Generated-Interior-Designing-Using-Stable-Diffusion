from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image

# Load your fine-tuned model
model_id = "my-ctrlnet"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# Load and process the input image
image = np.array(Image.open('bed1.jpg'))

# Canny edge detection
low_threshold = 150
high_threshold = 250
image = cv2.Canny(image, low_threshold, high_threshold)

# Convert Canny output to RGB format
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)  # Convert grayscale to RGB
canny_image = Image.fromarray(image)

# Use the fine-tuned model for inference
output = pipeline(
    prompt="Add a bed on side ,red bed sheet pattern, whitw curtain and few plant",  # Your prompt
    image=canny_image,  # The processed canny image
    negative_prompt="",  # Optional negative prompt
    num_inference_steps=16,  # Number of inference steps
).images[0]

# Save the output image
output.save('output_image_bed_new_img.png')
