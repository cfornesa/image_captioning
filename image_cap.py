import requests
from PIL import Image
# Process text and images to generate caption or answer questions
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pretrained BLIP model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = "Mapping AI Risk.jpeg"

# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"

# Process the image contents (as Pytorch tensors or "pt")
   # Encoding process
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image, max length of 50 tokens
    # Unpack dictionaries and pass items in the dict as keyword args to the model
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
    # Ignore special tokens in the output text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print(caption)