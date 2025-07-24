from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import requests
from io import BytesIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_url = "https://thumbs.dreamstime.com/b/velho-com-dor-nas-costas-av%C3%B4-s%C3%AAnior-lombar-fundo-branco-isolado-174152117.jpg"
response = requests.get(img_url)
raw_image = Image.open(BytesIO(response.content)).convert("RGB")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

question = "Is there a person in the image?"

inputs = processor(raw_image, question, return_tensors="pt").to(device)
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

raw_image.show()  # Isso abre a imagem com o visualizador do seu sistema
print(f"==> {answer}")