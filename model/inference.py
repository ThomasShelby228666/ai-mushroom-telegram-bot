import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import io
from typing import Union, Tuple

class MushroomPredictor:
    """
    Класс для предсказания видов грибов на основе их изображений с помощью модели EfficientNet (на базе CNN).
    Загружает предобученную модель и конфигурацию классов, далее выполняется инференс и возвращается предсказанный класс с уверенностью в предсказании.
    Attributes:
        device (torch.device): Устройство для вычислений (CUDA или CPU)
        classes (list): Список названий классов
        model (nn.Module): Загруженная модель
        transform (transforms.Compose): Преобразования для входных изображений
    """
    def __init__(self, model_path: str = 'model/mushroom_model.pth', config_path: str = 'model/config.json'):
        """
        Инициализирует предсказатель грибов.
        Загружает конфигурацию классов и веса модели, настраивает преобразования изображений.
        Args:
            model_path (str, optional): Путь к файлу с весами модели.
            config_path (str, optional): Путь к JSON-файлу с названиями классов.
        Raises:
            FileNotFoundError: Если файл конфигурации или файл модели не найден.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка конфигурации классов
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Нет файла {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)

        # Инициализация модели
        self.model = models.efficientnet_b0(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))

        # Загрузка весов модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Нет файла {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Настройка преобразований изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_input: Union[str, bytes, io.IOBase]) -> Tuple[str, float]:
        """
        Выполняет предсказание класса гриба по изображению.
        Args:
            image_input: Изображение для классификации.
        Returns:
            Tuple[str, float]: Кортеж, содержащий:
                - Название предсказанного класса (str)
                - Уверенность предсказания (float) в диапазоне [0, 1]
        """
        image = Image.open(image_input).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_p, top_class_idx = probs.topk(1, dim=1)
            
            return self.classes[top_class_idx.item()], top_p.item()