from supportbot.config import make_client


async def llm(
     messages: list[dict[str, str]],
     model: str = "qwen3:8b",
     temperature: float = 0.7,
     max_tokens: int = 2000,
 ) -> str:
     """
     Llama al modelo de lenguaje (LLM) especificado con el prompt dado y devuelve la respuesta generada.
 
     Args:
         messages (list[dict[str, str]]): La lista de mensajes para el modelo.
         model (str): El nombre del modelo a utilizar.
         temperature (float, optional): Controla la aleatoriedad de la salida. Valor por defecto es 0.7.
         max_tokens (int, optional): Número máximo de tokens en la respuesta. Valor por defecto es 512.
 
     Returns:
         str: La respuesta generada por el modelo.
 
     Raises:
         Exception: Si ocurre un error durante la llamada al LLM.
     """
     client,model = make_client("local")
     async with client:
        try:
            GRAY, RESET = "\033[90m", "\033[0m"
            # Cambiar "local" por el proveedor deseado según la configuración.
            # Aquí iría la lógica para llamar al LLM usando el proveedor y modelo especificados.
            # Esto puede incluir la construcción de la solicitud HTTP, manejo de autenticación, etc.
            response = ""
            stream = await client.chat.completions.create(model=model, messages=messages,max_tokens=max_tokens,temperature=temperature, stream=True)
            async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if r := getattr(delta, "reasoning_content", None):
                        print(f"{GRAY}{r}{RESET}", end="", flush=True)
                        response += r
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        response += delta.content
            print()   # o msg.model_extra.get("reasoning_content")
            return response
        except Exception as e:
            # Manejo de errores y logging
            raise Exception(f"Error calling LLM: {e}") from e