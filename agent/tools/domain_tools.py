# agent/tools/domain_tools.py

from __future__ import annotations

from langchain.tools import tool

from agent.llm_config import get_gemini_chat


@tool
def tarea_dominio_llm(instruccion: str) -> str:
    """
    Realiza una tarea de razonamiento en el dominio de vehículos y GANs
    usando un LLM (Gemini, texto).

    Ejemplos de uso:
    - Explicar cómo podríamos usar las imágenes generadas por la GAN
      en una aplicación real.
    - Proponer métricas cualitativas para evaluar la calidad de las
      imágenes de vehículos generadas.
    - Describir ventajas y limitaciones de usar GANs para este dominio.

    Parámetros:
    - instruccion: descripción en lenguaje natural de la tarea de dominio.

    Devuelve:
    - Respuesta en texto en español, bien explicada.
    """
    chat = get_gemini_chat(temperature=0.4)
    prompt = f"""
Eres un experto en vehículos y en modelos generativos (GANs).
Responde en español, de forma clara y bien estructurada.

Tarea solicitada:
{instruccion}
"""
    resp = chat.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


@tool
def recomendar_aplicacion_vehiculo_llm(descripcion: str) -> str:
    """
    Recomienda aplicaciones posibles para un vehículo descrito por el usuario.

    Este vehículo puede provenir de una imagen generada por la GAN
    (por ejemplo: 'camión grande de carga', 'moto pequeña urbana', etc.).

    Parámetros:
    - descripcion: descripción textual del vehículo.

    Devuelve:
    - Recomendaciones de uso (por ejemplo: transporte urbano, carga pesada,
      uso recreativo, flotas de reparto, etc.) en español.
    """
    chat = get_gemini_chat(temperature=0.5)
    prompt = f"""
Eres un experto en vehículos y aplicaciones de transporte.

Con base en la siguiente descripción de un vehículo generado por una IA:

\"\"\"{descripcion}\"\"\"

Responde en español:
1. ¿Qué tipo de aplicaciones o escenarios de uso serían adecuados para este vehículo?
2. ¿En qué contextos sería más útil (ciudad, carretera, industria, logística, etc.)?
3. Alguna limitación o consideración importante.

Escribe la respuesta en párrafos claros, sin listas ni JSON.
"""
    resp = chat.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
