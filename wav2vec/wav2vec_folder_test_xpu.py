"""
Тест классификатора Wav2Vec2 на папках с mp3.
Нативный PyTorch XPU (torch >= 2.5), intel_extension_for_pytorch не нужен.

Установка PyTorch с XPU:
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/xpu
"""

import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
MODEL_PATH = "wav2vec2_finetuned"

FRAME_RATE = 8000      # частота, до которой сжимаем mp3 (как при обучении)
CHUNK_SIZE = 24000     # длина куска в сэмплах (uint8, 1 байт на сэмпл)
TARGET_SR = 16000      # sampling_rate, который ждёт процессор
N = 3                  # число классов (артистов)
L = 40                 # число случайных кусков на файл
INFER_BATCH = 20       # размер батча при инференсе (уменьшите, если не хватает памяти)

USE_BF16 = False       # bfloat16 через autocast: быстрее и экономнее по памяти

# ---------------------------------------------------------------------------
# Устройство: XPU -> CUDA -> CPU
# ---------------------------------------------------------------------------
def pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


device = pick_device()
print("device:", device)
if device.type == "xpu":
    print("XPU:", torch.xpu.get_device_name(0))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if device.type == "xpu":
    torch.xpu.manual_seed_all(0)
elif device.type == "cuda":
    torch.cuda.manual_seed_all(0)

rnd = random.Random(0)

# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def get_random_batch_from_file(fn_in_mp3):
    """Нарезает L случайных кусков из mp3 и возвращает тензор (L, CHUNK_SIZE)."""
    sound = AudioSegment.from_mp3(fn_in_mp3)
    sound = sound.set_frame_rate(FRAME_RATE).set_channels(1).set_sample_width(1)

    raw = sound.raw_data
    total_len = len(raw)
    print(f"{total_len / FRAME_RATE:.1f} сек")

    # Защита от бесконечного цикла на слишком коротких файлах
    if total_len < 4 * CHUNK_SIZE:
        raise ValueError(f"Файл слишком короткий: {fn_in_mp3}")

    chunks = []
    while len(chunks) < L:
        start = round((0.9 * rnd.random() + 0.05) * total_len)
        if total_len - start < 3 * CHUNK_SIZE:
            continue
        chunk = np.frombuffer(raw[start:start + CHUNK_SIZE], dtype=np.uint8)
        chunk = chunk.astype(np.float32).reshape(1, CHUNK_SIZE)
        inputs = processor(
            chunk,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="max_length",
            max_length=CHUNK_SIZE,
            truncation=True,
        )
        chunks.append(inputs["input_values"].squeeze(0))

    return torch.stack(chunks).float()


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------
@torch.inference_mode()
def detect_file_type(fn_in_mp3):
    all_input_values = get_random_batch_from_file(fn_in_mp3)

    probs = []
    for i in range(0, len(all_input_values), INFER_BATCH):
        batch = all_input_values[i:i + INFER_BATCH].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=USE_BF16 and device.type != "cpu"):
            logits = model(input_values=batch).logits
        probs.append(torch.softmax(logits.float(), dim=1).cpu())

    mean_probs = torch.cat(probs).mean(dim=0).numpy() * 100

    for i in range(N):
        print(f"artist {i + 1}: {mean_probs[i]:.2f}")
    print(int(np.argmax(mean_probs)), float(np.max(mean_probs)))

    if device.type == "xpu":
        torch.xpu.empty_cache()

    return int(np.argmax(mean_probs)), float(np.max(mean_probs))


def do_it(folder, artist):
    result = np.zeros(N, dtype=int)
    fn_in_list = (glob.glob(os.path.join(folder, "*.mp3"))
                  + glob.glob(os.path.join(folder, "*", "*.mp3")))
    for i, fn in enumerate(fn_in_list):
        print(i, fn)
        try:
            pred, _ = detect_file_type(fn)
        except Exception as e:
            print(f"Пропускаю {fn}: {e}")
            continue
        result[pred] += 1
    return result


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 2:
        folder = sys.argv[1]
        artist = int(sys.argv[2])
        print(do_it(folder, artist))
    else:
        folders = [
            "N:\\Немного Нервно MP3",
            "N:\\music\\MP3\\The Cranberries",
            "N:\\music\\MP3\\Любэ",
        ]
        result = np.array([do_it(folders[a], a) for a in range(N)])
        print(result)

        plt.figure(figsize=(10, 8))
        sns.heatmap(result, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[str(i) for i in range(N)],
                    yticklabels=[str(i) for i in range(N)])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix_folder_test.png", dpi=150)
        plt.close()
