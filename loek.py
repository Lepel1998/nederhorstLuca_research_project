
# dit is copy paste van https://huggingface.co/google/vit-base-patch16-224-in21k

# er staat op huggingface altijd een voorbeeldje bij hoe je het model kan gebruiken
# maar je zult soms een beetje met de python help functie moeten kunnen

# tippie: doe dit niet in een python file maar jupyter notebook file

# probeer niet alleen deze ViT, maar meeredere, kijk welke het beste werkt!
# check: https://huggingface.co/models?pipeline_tag=image-feature-extraction&sort=trending

# maak je geen zorgen over computational resources, je hoeft niks te trainen dus
# dat is verwaarloosbaar! Als je 1 fototje kan draaien weet je dat 2000 ook wel gaan lukken

# doe even een pippie install transformers PIL requests
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# nu pakken we even random fotoje van internet, maar dit is dan een kever van je ofzo
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# met dit ding kan je de foto even in het goede format zetten
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# dit model is gewoon een PyTorch module! 
# Dus is eigenlijk zo'n Sequential ding (soort van, idee is hetzelfde) 
# maar dan door iemand anders gemaakt. 
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# nu is de foto het goede format. 
# Je kan hier ook meteen meerdere fotos in stoppen!
# Denk niet meteen alle 2000, kijk maar even wat je computer aankan :)
inputs = processor(images=image, return_tensors="pt")

# torch.Size([1, 3, 224, 224]) -> batch size, channels (RGB ofzo), height, width
print(inputs.pixel_values.shape) 

# en nu kunnen we de foto door het model heen halen
outputs = model(**inputs)

# dit heb ik gedaan om uit te zoeken wat er allemaal uit komt!
# kan je ook met bijvoorbeeld inputs doen!
# of als je durft kan je in vscode 
# (klik q om uit dat scherm te komen haha)

# print(help(outputs))
# print(dir(outputs))

# de output van de laatste laag
# torch.Size([1, 197, 768]) 
# -> 
# batch size, 
# aantal hokjes waarin de foto is opgedeeld (afhankelijk van de grootte dus), 
# aantal features
print(outputs.last_hidden_state.shape)

# dit is denk ik degene waarvan de bedoeling is dat je ze
# als 'features' gebruikt, ipv je manual features
# torch.Size([1, 768]) -> batch size, aantal features
print(outputs.pooler_output.shape)

# TODO

# 1. Haal al je ~2000 fotos door dit model heen en sla de pooler outputs op!
# 2. Je hebt nu voor elke foto een vector van 768 features
# 3. En je hebt voor elke foto een label/class/insectsoort/whatever
# 4. Train een simpel model, zoals een SVM/NB met deze features en labels
# 5. Kijk of het beter werkt dan je manual features
# 6. Doe hetzelfde voor een andere feature extractor die je op huggingface hebt gevonden
# 7. Succes :D