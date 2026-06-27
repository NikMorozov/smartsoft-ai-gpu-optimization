import os
import time
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms


IMAGES_DIR = "images"
RESULTS_FILE = "results.csv"


def get_device():
    """
    Проверяет доступность GPU.
    Если видеокарта доступна, используется CUDA.
    Если нет — программа работает на CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    """
    Загружает предобученную модель MobileNetV2
    и переносит ее на выбранное устройство.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.to(device)
    model.eval()
    return model


def prepare_image(image_path):
    """
    Подготавливает изображение для обработки нейронной сетью.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def classify_images():
    device = get_device()
    print(f"Используемое устройство: {device}")

    model = load_model(device)

    results = []
    start_time = time.time()

    if not os.path.exists(IMAGES_DIR):
        print("Папка images не найдена. Создайте папку и добавьте изображения.")
        return

    image_files = [
        file for file in os.listdir(IMAGES_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("В папке images нет изображений для обработки.")
        return

    with torch.no_grad():
        for file_name in image_files:
            image_path = os.path.join(IMAGES_DIR, file_name)

            try:
                image = prepare_image(image_path)
                image = image.to(device)

                output = model(image)
                predicted_class = torch.argmax(output, dim=1).item()

                results.append({
                    "file_name": file_name,
                    "predicted_class": predicted_class,
                    "device": str(device)
                })

                print(f"Обработано: {file_name}")

            except Exception as error:
                print(f"Ошибка при обработке {file_name}: {error}")

    end_time = time.time()

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)

    print(f"Результаты сохранены в файл {RESULTS_FILE}")
    print(f"Время обработки: {round(end_time - start_time, 2)} сек.")


if __name__ == "__main__":
    classify_images()
