import asyncio
import logging
import os
import sys
import io
# Отключение создания .pyc файлов
sys.dont_write_bytecode = True
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

# Добавляем путь для импорта локальных модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.inference import MushroomPredictor
from services.llm import get_mushroom_info


# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Инициализация классификатора грибов
try:
    print("Инициализация нейросети...")
    predictor = MushroomPredictor()
except Exception as e:
    print(f" Фатальная ошибка: {e}")
    sys.exit(1)

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message) -> None:
    """
    Обработчик команды /start.
    Отправляет приветственное сообщение с инструкцией по использованию бота.
    Args:
        message (types.Message): Объект сообщения от пользователя
    """
    await message.answer("📸 **Отправь мне фото гриба!**\nА я определю его вид и съедобность.")

@dp.message(F.photo)
async def handle_photo(message: types.Message) -> None:
    """
    Обработчик фотографий, присланных пользователем.
    Выполняет следующие действия:
    1. Отправляет статус "печатает..."
    2. Загружает фотографию
    3. Классифицирует гриб с помощью нейросети
    4. Генерирует описание через LLM
    5. Отправляет результат пользователю
    Args:
        message (types.Message): Объект сообщения с фотографией
    """
    # Показываем статус "печатает..."
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Отправляем начальное сообщение о обработке
    status_msg = await message.answer("**Обрабатываю...**")
    
    try:
        # Загружаем фотографию
        buffer = io.BytesIO()
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        await bot.download_file(file.file_path, destination=buffer)
        buffer.seek(0)

        # Классифицируем гриб
        class_name, prob = predictor.predict(buffer)
        confidence = prob * 100

        # Обновляем статус
        await status_msg.edit_text(f"🍄‍🟫 Это **{class_name}** ({confidence:.1f}%)\n⏳ Генерирую описание...")
        desc = await get_mushroom_info(class_name, confidence)

        # Получаем описание от LLM
        await status_msg.edit_text(
            f"Вид: **{class_name}**\n"
            f"Точность: `{confidence:.1f}%`\n\n"
            f"{desc}"
        )

    except Exception as e:
        # Логируем ошибку и сообщаем пользователю
        logging.error(f"Ошибка: {e}")
        await status_msg.edit_text("Не удалось обработать фото.")


async def main() -> None:
    """
    Главная асинхронная функция для запуска бота.
    """
    print("Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Запуск бота
    asyncio.run(main())