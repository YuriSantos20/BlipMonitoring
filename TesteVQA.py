from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_path = r"C:\Users\Admin\Downloads\ClassificaçãoGeralEmoção-20250712T160954Z-1-001\ClassificaçãoGeralEmoção\SemDor\imagem_2.jfif"
raw_image = Image.open(img_path).convert("RGB")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

question = "Is there a person in the image?"

inputs = processor(raw_image, question, return_tensors="pt").to(device)
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

raw_image.show()  # Isso abre a imagem com o visualizador do seu sistema
print(f"==> {answer}")