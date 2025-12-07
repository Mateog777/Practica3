# agent/tools/gan_tools.py

from __future__ import annotations

from typing import List

from langchain.tools import tool

from gan.vehicle_gan_inference import generate_vehicle_images


@tool
def generar_imagen_gan(modelo: str = "best", num_imagenes: int = 4) -> List[str]:
    """
    Genera imágenes sintéticas usando la GAN entrenada de vehículos.

    Parámetros:
    - modelo: "best" para el mejor checkpoint o "final" para el último checkpoint.
    - num_imagenes: número de imágenes a generar (1 a 16).

    Devuelve:
    - Lista de rutas (strings) de imágenes PNG generadas.
    """
    modelo = (modelo or "best").strip().lower()
    if modelo not in {"best", "final"}:
        modelo = "best"

    num_imagenes = max(1, min(int(num_imagenes), 16))

    return generate_vehicle_images(kind=modelo, num_images=num_imagenes)

