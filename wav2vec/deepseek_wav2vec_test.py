"""
Complete Audio Classification Pipeline with Wav2Vec2
Single File Solution
"""

import os
import sys
import warnings
import tempfile
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.xpu.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(6)

# torch.set_default_device('cpu')
# torch.set_default_dtype(torch.float32)

audio_data_raw_dir = f'../audio_data_raw'

artist_count = 3

samples_per_file = 300

files_per_artist_train = 18
files_per_artist_validate = 6

files_per_artist_total = files_per_artist_train + files_per_artist_validate

# Transformers
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate

# Отключение предупреждений
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 1. СОЗДАНИЕ ДАТАСЕТА ИЗ ПАПОК
# ============================================================================

# ============================================================================
# 2. ДАТАСЕТ ДЛЯ WAV2VEC2
# ============================================================================


class Wav2Vec2Dataset(Dataset):
    """Датасет для fine-tuning Wav2Vec2"""
    
    def __init__(self, processor, dataset_name):
        self.processor = processor
        self.dataset_name = dataset_name
        self.sample_len = 24000
        with open(f'{audio_data_raw_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_{dataset_name}.raw',
                  "rb") as file:
            raw_artist_data = file.read()
        self.artist_data = np.frombuffer(raw_artist_data, dtype=np.uint8).astype(float)
        (H, ) = self.artist_data.shape
        self.sample_count = H // self.sample_len
        self.artist_data = np.reshape(self.artist_data, (self.sample_count, self.sample_len))
        self.target_sr = 16000
        
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, idx):
        try:
            label = (artist_count * idx) // self.sample_count

            # data = np.pad(self.artist_data[idx, :], (80000 - 24000) // 2, 'constant', constant_values=0)
            # data = np.reshape(data, (1, 80000))
            data = self.artist_data[idx, :]
            data = np.reshape(data, (1, 24000))
            # print(data.shape)

            inputs = self.processor(
                data,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding="max_length",
                max_length=24000,
                truncation=True
            )

            input_values = inputs['input_values'].squeeze(0)

            # print(input_values.shape)

            return {
                'input_values': input_values,
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            # В случае ошибки возвращаем нулевой тензор
            print(f"Ошибка при обработке сэмпла {idx}: {e}")

# ============================================================================
# 3. FINE-TUNING WAV2VEC2
# ============================================================================


def finetune_wav2vec2(num_classes, output_dir="wav2vec2_finetuned"):
    """
    Fine-tuning модели Wav2Vec2
    
    Args:
        train_df: DataFrame тренировочных данных
        val_df: DataFrame валидационных данных
        num_classes: количество классов
        output_dir: директория для сохранения модели
    """
    print("=" * 60)
    print("FINE-TUNING WAV2VEC2")
    print("=" * 60)
    
    # Создаем директорию для модели
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем процессор
    print("Загрузка Wav2Vec2 процессора...")
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    except:
        # Если нет интернета, используем локальную копию
        print("Использую локальную версию процессора...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)

    # Создаем датасеты
    print("Создание датасетов...")
    train_dataset = Wav2Vec2Dataset(processor, 'train')
    val_dataset = Wav2Vec2Dataset(processor, 'validate')

    print(f"Размер train датасета: {len(train_dataset)}")
    print(f"Размер val датасета: {len(val_dataset)}")

    # Загружаем модель
    print("Загрузка модели Wav2Vec2...")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_classes,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            classifier_proj_size=256,
            ignore_mismatched_sizes=True
        ) #.cpu()
    except:
        print("Использую локальную версию модели...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        ) #.cpu()

    # Настраиваем заморозку слоев
    print("Настройка заморозки слоев...")

    # Вариант 1: Замораживаем все, кроме классификатора
    for name, param in model.named_parameters():
        if 'classifier' in name or 'projector' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Вариант 2: Размораживаем последние N слоев
    # Список слоев для разморозки
    # unfreeze_layers = ['encoder.layers.11.', 'encoder.layers.10.',
    #                   'encoder.layers.9.', 'encoder.layers.8.']
    #
    # for name, param in model.named_parameters():
    #     if any(layer in name for layer in unfreeze_layers):
    #         param.requires_grad = True

    # Считаем trainable параметры
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable параметры: {trainable_params:,} из {total_params:,} "
          f"({trainable_params/total_params*100:.1f}%)")

    # Метрики
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        remove_unused_columns=False,
        gradient_checkpointing=True if torch.cuda.is_available() else False,
        save_only_model=True,
        # use_cpu=True,
    )

    print('device', training_args.device)

    # Создаем Trainer
    print("Создание Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Обучение
    print("\nНачало обучения...")
    start_time = time.time()

    try:
        train_result = trainer.train()
        training_time = time.time() - start_time

        print(f"\n✅ Обучение завершено за {training_time:.1f} секунд")
        print(f"Final train loss: {train_result.training_loss:.4f}")

        # Сохраняем модель
        print("Сохранение модели...")
        trainer.save_model()
        processor.save_pretrained(output_dir)

        # Сохраняем результаты обучения
        with open(os.path.join(output_dir, "training_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)

        # Визуализация лосса
        if hasattr(trainer.state, 'log_history'):
            plot_training_history(trainer.state.log_history, output_dir)

        return model, processor

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_training_history(history, output_dir):
    """Визуализация истории обучения"""
    try:
        train_loss = [x['loss'] for x in history if 'loss' in x]
        eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
        eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(train_loss, label='Train Loss', marker='o')
        if eval_loss:
            axes[0].plot(eval_loss, label='Eval Loss', marker='s')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if eval_acc:
            axes[1].plot(eval_acc, label='Eval Accuracy', marker='s', color='green')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
        plt.close()
        print("📊 Графики обучения сохранены")
        
    except Exception as e:
        print(f"Ошибка при построении графиков: {e}")

# ============================================================================
# 4. ТЕСТИРОВАНИЕ МОДЕЛИ
# ============================================================================


def test_model(model, processor, test_df, class_mapping, batch_size=8):
    """
    Тестирование обученной модели
    
    Args:
        model: обученная модель
        processor: Wav2Vec2Processor
        test_df: DataFrame тестовых данных
        class_mapping: словарь маппинга классов
        batch_size: размер батча для инференса
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("=" * 60)
    
    # Создаем датасет
    test_dataset = Wav2Vec2Dataset(processor, 'validate')
    
    # DataLoader для батчинга
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Переводим модель в eval режим
    model.eval()
    device = next(model.parameters()).device
    print(f"Используемое устройство: {device}")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Инференс
    print("Запуск инференса...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            # Перемещаем данные на устройство
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Предсказания
            outputs = model(input_values=input_values)
            logits = outputs.logits
            
            # Вероятности
            probabilities = torch.softmax(logits, dim=1)
            
            # Предсказанные классы
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Конвертируем в numpy массивы
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Вычисляем метрики
    accuracy = np.mean(all_predictions == all_labels)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"Точность (accuracy): {accuracy:.4f}")
    print(f"Количество образцов: {len(test_df)}")
    
    # Отчет по классификации
    print("\n📋 ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
    target_names = [class_mapping.get(i, f"Class_{i}") for i in range(len(class_mapping))]
    
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=target_names,
        digits=3
    )
    print(report)
    
    # Матрица ошибок
    print("📈 МАТРИЦА ОШИБОК:")
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Матрица ошибок сохранена как 'confusion_matrix.png'")
    
    # Сохраняем результаты
    results = {
        'accuracy': float(accuracy),
        'total_samples': len(test_df),
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
        'classification_report': report
    }
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    
    return accuracy, all_predictions, all_probabilities


def get_class_name(label, class_mapping):
    """Универсальное получение имени класса"""
    # Пробуем разные варианты ключей
    if label in class_mapping:
        return class_mapping[label]
    elif str(label) in class_mapping:
        return class_mapping[str(label)]
    elif int(label) in class_mapping:
        return class_mapping[int(label)]
    else:
        return f"Class_{label}"


def model_exists(model_dir):
    """Проверяет, существует ли обученная модель"""
    # Новые названия файлов моделей в transformers
    model_files = [
        "pytorch_model.bin",  # Старое название
        "model.safetensors",  # Новое название (по умолчанию)
        "pytorch_model.bin.index.json",  # Для больших моделей
        "model-00001-of-00001.safetensors",  # Для шардированных моделей
    ]

    # Проверяем наличие хотя бы одного файла модели
    for file in model_files:
        if os.path.exists(os.path.join(model_dir, file)):
            return True

    # Или проверяем по наличию config.json
    if (os.path.exists(os.path.join(model_dir, "config.json")) and
            os.path.exists(os.path.join(model_dir, "preprocessor_config.json"))):
        return True

    return False


# ============================================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция пайплайна"""
    print("=" * 60)
    print("АУДИО КЛАССИФИКАТОР НА WAV2VEC2")
    print("=" * 60)
    
    # Параметры
    MODEL_DIR = "wav2vec2_finetuned"  # Папка для модели
    
    print(f"\nКоличество классов: {artist_count}")
    
    # 2. Fine-tuning
    print("\n2. FINE-TUNING МОДЕЛИ")

    if model_exists(MODEL_DIR):
        print("Модель уже обучена, пропускаем обучение...")
        model = None
        processor = None
    else:
        model, processor = finetune_wav2vec2(
            artist_count, MODEL_DIR
        )

        if model is None:
            print("❌ Обучение не удалось!")
            return
    
    # 3. Тестирование
    print("\n3. ТЕСТИРОВАНИЕ МОДЕЛИ")
    
    # Загружаем модель если она не была обучена сейчас
    if model is None:
        try:
            processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return
    
    print("\n" + "=" * 60)
    print("ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("=" * 60)
    print("\nСозданные файлы:")
    print("  data_splits/ - разделенные данные")
    print("  wav2vec2_finetuned/ - обученная модель")
    print("  test_results.json - результаты тестирования")
    print("  confusion_matrix.png - матрица ошибок")
    
    if os.path.exists("training_history.png"):
        print("  training_history.png - графики обучения")

# ============================================================================
# 7. УТИЛИТЫ
# ============================================================================

# def create_test_data():
#     """Создание тестовых данных для демонстрации"""
#     print("Создание тестовых данных...")
#
#     # Создаем тестовую структуру
#     test_dir = "test_audio_data"
#     os.makedirs(os.path.join(test_dir, "speech"), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, "music"), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, "noise"), exist_ok=True)
#
#     import wave
#     import numpy as np
#
#     for i in range(10):
#         for class_name, frequency in [("speech", 440), ("music", 523), ("noise", 659)]:
#             file_path = os.path.join(test_dir, class_name, f"{class_name}_{i}.wav")
#
#             # Создаем простой аудио сигнал
#             sample_rate = 16000
#             duration = 2.0
#             t = np.linspace(0, duration, int(sample_rate * duration))
#
#             if class_name == "noise":
#                 audio = np.random.normal(0, 0.1, len(t))
#             else:
#                 audio = 0.5 * np.sin(2 * np.pi * frequency * t)
#
#             # Добавляем затухание
#             envelope = np.exp(-t)
#             audio = audio * envelope
#
#             # Нормализация и конвертация в int16
#             audio = audio / np.max(np.abs(audio))
#             audio = (audio * 32767).astype(np.int16)
#
#             # Сохраняем как WAV
#             with wave.open(file_path, 'wb') as wav_file:
#                 wav_file.setnchannels(1)
#                 wav_file.setsampwidth(2)
#                 wav_file.setframerate(sample_rate)
#                 wav_file.writeframes(audio.tobytes())
#
#     print(f"Тестовые данные созданы в папке {test_dir}")
#     return test_dir

# ============================================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================================


if __name__ == "__main__":
    # Если нет данных, создаем тестовые
    # if not os.path.exists("audio_data") and not os.path.exists("data_splits/train.csv"):
    #     print("Тестовые данные не найдены.")
    #     create_test = input("Создать тестовые данные? (y/n): ")
    #     if create_test.lower() == 'y':
    #         DATA_DIR = create_test_data()
    #         print(f"\nЗапускайте программу с DATA_DIR = '{DATA_DIR}'")
    
    # Запуск основного пайплайна
    main()
