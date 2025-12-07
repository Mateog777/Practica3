# agent/tools/vision_tools.py

from __future__ import annotations

import os
from typing import Dict

from langchain.tools import tool
from PIL import Image
import google.generativeai as genai

from config.settings import GOOGLE_API_KEY, GEMINI_MODEL_NAME


if not GOOGLE_API_KEY:
    raise ValueError("Falta GOOGLE_API_KEY en el .env para usar Gemini Vision")

genai.configure(api_key=GOOGLE_API_KEY)


def _load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return Image.open(path)


@tool
def analizar_imagen_llm(ruta_imagen: str) -> str:
    """
    Analiza una imagen de vehículo generada por la GAN usando Gemini Vision.

    Parámetros:
    - ruta_imagen: ruta en disco de la imagen (por ejemplo, una generada por la GAN).

    Devuelve:
    - Un análisis en texto en español describiendo:
      * tipo aproximado de vehículo,
      * características visuales importantes,
      * posibles problemas o artefactos generados por la GAN.
    """
    img = _load_image(ruta_imagen)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    prompt = """
Eres un experto analizando imágenes de vehículos generadas por una red generativa (GAN).

Analiza la imagen y responde en español, explicando:

1. Qué tipo de vehículo parece (auto, moto, camión, bus, bicicleta, etc.).
2. Qué características visuales se aprecian (forma, ruedas, luces, ventanas, etc.).
3. Si ves artefactos típicos de GAN (bordes raros, formas deformadas, ruido, etc.).
4. Una conclusión breve sobre qué tan realista o útil es la imagen para el dominio de vehículos.

Escribe tu respuesta en formato de párrafos claros, sin listas ni JSON.
"""

    response = model.generate_content([prompt, img])
    return response.text or ""


@tool
def comparar_imagenes_vehiculos_llm(
    ruta_imagen_1: str,
    ruta_imagen_2: str,
) -> str:
    """
    Compara dos imágenes de vehículos generadas (por la GAN o reales) usando Gemini Vision.

    Parámetros:
    - ruta_imagen_1: ruta de la primera imagen.
    - ruta_imagen_2: ruta de la segunda imagen.

    Devuelve:
    - Un análisis comparativo en español:
      * similitudes entre los vehículos,
      * diferencias (tipo, tamaño, forma, detalles),
      * cuál parece más realista y por qué.
    """
    img1 = _load_image(ruta_imagen_1)
    img2 = _load_image(ruta_imagen_2)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    prompt = """
Eres un experto en análisis de imágenes de vehículos.

Compara LAS DOS imágenes que te paso y responde en español:

1. ¿Qué tipo de vehículo parece haber en cada imagen?
2. ¿En qué se parecen (forma, color, tipo de vehículo, estilo, etc.)?
3. ¿En qué se diferencian claramente?
4. ¿Cuál de las dos se ve más realista o mejor lograda, y por qué?

Escribe tu respuesta en uno o varios párrafos, sin listas ni JSON.
"""

    response = model.generate_content([prompt, img1, img2])
    return response.text or ""
