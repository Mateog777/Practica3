# agent/agent_runner.py

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm_config import get_gemini_chat
from agent.tools import ALL_TOOLS


SYSTEM_PROMPT = """
Eres un asistente experto en imágenes de vehículos generadas por una GAN
y en modelos de lenguaje aplicados al dominio de vehículos.

Herramientas que puedes usar:

- generar_imagen_gan: genera imágenes de vehículos usando una GAN entrenada.
- analizar_imagen_llm: analiza una imagen de vehículo usando Gemini Vision.
- tarea_dominio_llm: realiza tareas de razonamiento en texto sobre el dominio de vehículos y GANs.
- comparar_imagenes_vehiculos_llm: compara dos imágenes de vehículos y explica similitudes/diferencias.
- recomendar_aplicacion_vehiculo_llm: recomienda aplicaciones/proyectos según la descripción de un vehículo.

Debes:
- Decidir cuándo llamar a una herramienta si ayuda a responder mejor.
- Explicar SIEMPRE en español.
- Cuando uses herramientas de imagen, di al usuario qué hiciste y qué encontraste.
"""


def build_agent():
    """
    Construye un agente básico con Gemini + tools (GAN de vehículos, etc.).
    """
    model = get_gemini_chat(temperature=0.3)
    agent = create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_once(pregunta: str) -> str:
    """
    Ejecuta una sola interacción con el agente.
    """
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=pregunta),
            ]
        }
    )

    # En muchas versiones de LangChain, el resultado viene como dict con 'output'
    if isinstance(result, dict) and "output" in result:
        return str(result["output"])
    return str(result)


if __name__ == "__main__":
    print("Prueba rápida del agente de vehículos 🚗")
    respuesta = run_once(
        "Genera 3 imágenes de vehículos con la herramienta adecuada "
        "y dime qué tipos de vehículos parecen."
    )
    print("\nRespuesta del agente:\n", respuesta)
