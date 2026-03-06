import os
import logging
from openai import AsyncOpenAI 
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Инициализация асинхронного клиента OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Название модели для использования
MODEL_NAME = os.getenv("LLM_MODEL")

def load_prompt(filename: str = "prompt.txt") -> str:
    """
    Загружает системный промпт из текстового файла.
    Args:
        filename (str, optional): Имя файла с промптом.
    Returns:
        str: Содержимое файла промпта.
    Raises:
        FileNotFoundError: Если файл не найден по указанному пути.
        IOError: При ошибках чтения файла.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Загрузка системного промпта при инициализации модуля
SYSTEM_PROMPT = load_prompt()

async def get_mushroom_info(mushroom_name: str, confidence: float) -> str:
    """
        Асинхронно получает информацию о грибе от LLM.
        Формирует запрос к OpenRouter API с системным промптом и пользовательским запросом.
        Args:
            mushroom_name (str): Название гриба, распознанное классификатором.
            confidence (float): Уверенность распознавания в процентах (0-100).
        Returns:
            str: Текстовое описание гриба от LLM или сообщение об ошибке.
        """
    user_prompt = (
        f"Нейросеть распознала на фото гриб: **{mushroom_name}**.\n"
        f"Уверенность распознавания: {confidence:.1f}%.\n"
        "Дай справку по этому грибу."
    )

    print(f" Отправляю запрос к LLM ({MODEL_NAME}) для гриба {mushroom_name}...")

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=20.0
        )

        print(" Ответ от LLM получен!")
        return response.choices[0].message.content

    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return ("Не могу получить описание от нейросети (таймаут или ошибка). "
            f"Но мой классификатор уверен, что это **{mushroom_name}**."
        )