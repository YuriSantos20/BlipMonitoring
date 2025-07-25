import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from lavis.models import load_model_and_preprocess

# Define dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelo VQA (Question Answering)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Carrega modelo ITM (Image-Text Matching)
model_itm, vis_processors_itm, text_processors_itm = load_model_and_preprocess(
    name="blip2_image_text_matching", 
    model_type="pretrain", 
    is_eval=True, 
    device=device
)

# Imagem de exemplo via URL
from io import BytesIO
import requests

img_url = "https://img.freepik.com/fotos-gratis/mulher-atraente-feliz-dancando-e-se-divertindo-levantando-as-maos-despreocupada-curtindo-musica-encostada-na-parede-branca_176420-38816.jpg"
response = requests.get(img_url)
raw_image = Image.open(BytesIO(response.content)).convert("RGB")

# Pergunta inicial
question = "Is there a person in the image?"
inputs = processor(raw_image, question, return_tensors="pt").to(device)
out = model_vqa.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

raw_image.show()
print(f"Resposta da VQA: {answer}")

# Define frases para ITM
frases = [
    'Person is in pain',
    'Person is angry',
    'Person is disgusted',
    'Person is happy',
    'Person is sad',
    'Person is surprised'
]

# Pré-processa textos
textos_itm = [text_processors_itm["eval"](frase) for frase in frases]


# Avaliação por ITM
if answer.lower() == 'yes':
    img_itm = vis_processors_itm["eval"](raw_image).unsqueeze(0).to(device)
    scores = []

    for txt in textos_itm:
        with torch.no_grad():
            itm_output = model_itm({"image": img_itm, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            scores.append(itm_scores[:, 1].item())

    print(f'Scores gerados: {scores}')
    maior = max(scores)
    h = scores.index(maior)
    print(f'Maior score: {maior:.4f}')
    print(f'Posição do maior: {h} => "{frases[h]}"')

    if h == 0:
        print('✅ Person is in pain')
        
    else:
        print('❌ Person is not in pain')
        
else:
    print("A imagem não contém uma pessoa. Ignorado.")

