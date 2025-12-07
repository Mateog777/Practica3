# agent/llm_config.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config.settings import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GEMINI_MODEL_NAME,
    GROQ_MODEL_NAME,
)


def get_gemini_chat(temperature: float = 0.2):
    """
    Devuelve un ChatModel de Gemini listo para usarse con LangChain.
    Usa convert_system_message_to_human para evitar problemas con
    versiones recientes de la API.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("Falta GOOGLE_API_KEY en el .env")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        api_key=GOOGLE_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def get_groq_chat(temperature: float = 0.2):
    """
    Opcional: modelo de Groq para otras tareas de texto.
    """
    if not GROQ_API_KEY:
        raise ValueError("Falta GROQ_API_KEY en el .env")

    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )
