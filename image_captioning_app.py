# Import:
    # gradio
    # numpy
    # PIL: Image
    # transformers: AutoProcessor and BlipForConditionalGeneration
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pretrained BLIP model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image caption function
def caption_image(input_image: np.ndarray):

    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image contents (as Pytorch tensors or "pt")
        # Encoding process
    inputs = processor(raw_image, return_tensors="pt")

    # Generate a caption for the image, max length of 50 tokens
        # Unpack dictionaries and pass items in the dict as keyword args to the model
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
        # Ignore special tokens in the output text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Create the web interface using Gradio
iface = gr.Interface(
    fn = caption_image, 
    inputs = gr.Image(), 
    outputs = "text", 
    title = "Image Captioning", 
    description = "This is a simple web app for generating captions for images using a trained model (BLIP from Salesforce via Hugging Face)."
)

# Launch the interface
iface.launch()