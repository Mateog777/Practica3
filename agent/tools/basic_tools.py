# agent/tools/basic_tools.py

from langchain.tools import tool


@tool
def eco(texto: str) -> str:
    """Devuelve exactamente el mismo texto que recibe. Útil para pruebas."""
    return texto


@tool
def sumar(a: float, b: float) -> float:
    """Suma dos números y devuelve el resultado."""
    return a + b
