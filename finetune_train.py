import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        # Collect both .jpg and .png files
        self.image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Load the corresponding prompt from the .txt file
        # Replace the extension with .txt, whether it's .jpg or .png
        if img_path.endswith(".jpg"):
            prompt_path = img_path.replace(".jpg", ".txt")
        elif img_path.endswith(".png"):
            prompt_path = img_path.replace(".png", ".txt")

        try:
            with open(prompt_path, "r", encoding='utf-8') as file:
                prompt = file.read().strip()
        except UnicodeDecodeError:
            print(f"Error reading the prompt file: {prompt_path}. Skipping this file.")
            prompt = ""  # Provide a default empty prompt or handle this case as necessary

        if self.transform:
            image = self.transform(image)

        return image, prompt

# Function to check if an image is black
def is_black_image(image_tensor):
    """
    Check if the image is black by calculating the sum of all pixel values.
    If the sum is zero, it's likely a black image.
    """
    return torch.sum(image_tensor) == 0

# Transformations
transform = transforms.Compose([
    transforms.Resize((256 , 256)),  # Resize to 126x126 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

image_folder = os.path.join(os.path.dirname(__file__), "Mydataset")
# Initialize the dataset and data loader
dataset = CustomDataset(image_folder=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=4 , shuffle=True, pin_memory=True)

# Load ControlNet and Stable Diffusion Pipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "medmac01/beldi-moroccan-interior-2", controlnet=controlnet, torch_dtype=torch.float16
)

# Enable optimizations
pipe.enable_model_cpu_offload()

# Tokenizer (optional if needed for your task)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Adjust as necessary

# Define a loss function
loss_fn = torch.nn.MSELoss()  # Mean Squared Error

# Store loss for visualization
losses = []

def train_fine_tune():
    num_epochs = 5  # Reduced from 5
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, prompts) in enumerate(dataloader):
            for j, (image, prompt) in enumerate(zip(images, prompts)):
                image = image.unsqueeze(0).to("cuda")
                pil_image = transforms.ToPILImage()(image.squeeze(0).cpu())

                output = pipe(
                    prompt=prompt,
                    image=pil_image,
                    negative_prompt = "nsfw, adult, nudity, inappropriate, dark tones, low resolution, distorted shapes, bad anatomy, worst quality, low quality",
                    num_inference_steps=20,  # Reduced from 10
                )

                output_image = output.images[0]
                output_image_tensor = transforms.ToTensor()(output_image).to("cuda")

                # Check if the image is black
                if is_black_image(output_image_tensor):
                    print(f"NSFW content detected for Batch [{i+1}], Image [{j+1}]. Skipping this image.")
                    continue  # Skip this image and move to the next one

                # Resize the output image tensor to match the input image tensor size
                output_image_tensor = torch.nn.functional.interpolate(
                    output_image_tensor.unsqueeze(0),  # Add batch dimension
                    size=image.shape[2:],  # Match the height and width of the input image
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove the batch dimension

                # Calculate loss
                loss = loss_fn(output_image_tensor, image)
                epoch_loss += loss.item()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Image [{j+1}] processed, Loss: {loss.item()}.")

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")

    pipe.save_pretrained("my-ctrlnet")

    plt.plot(range(num_epochs), losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    train_fine_tune()
