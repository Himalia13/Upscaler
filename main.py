import torch
print(torch.cuda.is_available())  # Esto debería devolver True si la GPU está disponible
print(torch.cuda.get_device_name(0))  # Muestra el nombre de tu GPU



import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image

# Detectar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargar el modelo
model_path = 'RealESRGAN_x4plus.pth'
state_dict = torch.load(model_path, map_location=device)['params_ema']

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)

# Configurar RealESRGANer con GPU
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=256,  # Ajusta según la memoria disponible (e.g., 128 o 512)
    tile_pad=10,
    pre_pad=0,
    half=True if device.type == 'cuda' else False,
    device=device
)

# Procesar el video
input_video = 'input_video.mp4'
output_video = 'output_video.mp4'

# Abrir el video de entrada
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Configurar el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width * 4, frame_height * 4))  # Multiplicado por 4 por la escala

# Procesar fotograma por fotograma
print(f'Procesando {frame_count} fotogramas...')

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a formato RGB requerido por Real-ESRGAN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Aplicar superresolución
    try:
        output, _ = upsampler.enhance(img_rgb, outscale=4)
    except RuntimeError as error:
        print(f"Error procesando el fotograma {i}: {error}")
        output = img_rgb  # Usa el fotograma original si hay un error

    # Convertir de nuevo a BGR para guardar en el video
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Escribir el fotograma procesado en el video de salida
    out.write(output_bgr)

    # Liberar memoria de la GPU
    torch.cuda.empty_cache()

    # Mensaje de progreso
    if i % 10 == 0 or i == frame_count - 1:
        print(f'Fotograma {i}/{frame_count} procesado ({(i / frame_count) * 100:.2f}%).')

# Liberar recursos
cap.release()
out.release()
print('Video procesado y guardado en:', output_video)
