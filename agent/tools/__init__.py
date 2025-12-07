# agent/tools/__init__.py

from .gan_tools import generar_imagen_gan
from .vision_tools import analizar_imagen_llm, comparar_imagenes_vehiculos_llm
from .domain_tools import tarea_dominio_llm, recomendar_aplicacion_vehiculo_llm

ALL_TOOLS = [
    generar_imagen_gan,
    analizar_imagen_llm,
    tarea_dominio_llm,
    comparar_imagenes_vehiculos_llm,
    recomendar_aplicacion_vehiculo_llm,
]
