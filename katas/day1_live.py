import asyncio
import supportbot

async def main():
    print("Iniciando prueba de LLM...")
    # Ejemplo de uso del LLM con un prompt simple
    messages = [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Hola, ¿puedes ayudarme con un problema de programación?"}
    ]


    
    await supportbot.llm.llm(messages)
    
if __name__ == "__main__":
    asyncio.run(main())