# main.py

from agent.agent_runner import run_once


def main():
    print("=== Agente de Vehículos con GAN + Gemini ===")
    while True:
        try:
            user_input = input("\nTu pregunta (o 'salir'): ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in {"salir", "exit", "quit"}:
            break

        respuesta = run_once(user_input)
        print("\n🧠 Agente:\n", respuesta)


if __name__ == "__main__":
    main()
