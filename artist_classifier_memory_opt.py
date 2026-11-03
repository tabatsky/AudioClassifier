import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from artist_net import make_actual_nn, sample_len, version_name

# ----------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------
artist_count = 3

samples_per_file = 1000
files_per_artist_train = 18
files_per_artist_validate = 6
files_per_artist_total = files_per_artist_train + files_per_artist_validate

count_per_artist_train = files_per_artist_train * samples_per_file
count_per_artist_validate = files_per_artist_validate * samples_per_file
count_train = count_per_artist_train * artist_count
count_validate = count_per_artist_validate * artist_count

LR = 1e-3
LR_GAMMA = 0.995          # StepLR(step_size=1) как раньше
BATCH_SIZE = 100
VAL_BATCH_SIZE = 500      # в eval нет градиентов, батч можно брать больше
END_EPOCH = 300           # последняя эпоха включительно (как в исходном скрипте)
PATIENCE = 0              # ранняя остановка: 0 = выключена, иначе число эпох без улучшения
RESUME = False            # продолжить с лучшего чекпоинта
USE_AMP = False           # bf16 autocast: меньше памяти на активации, но чуть менее точно
DATA_ON_DEVICE = True     # False: данные в RAM, на устройство едет только батч

working_dir = '.'
run_name = f'{version_name}_{artist_count}_{samples_per_file}_{files_per_artist_total}'
weights_dir = f'{working_dir}/weights/{run_name}_weights'
best_path = f'{weights_dir}/best.pth'
audio_data_raw_dir = f'{working_dir}/audio_data_raw'
log_path = f'{working_dir}/accuracy/{run_name}_metrics.csv'

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# ----------------------------------------------------------------------------
# Устройство и воспроизводимость
# ----------------------------------------------------------------------------
has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
print('cuda available:', torch.cuda.is_available())
print('xpu available:', has_xpu)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
if has_xpu:
    device = torch.device('xpu:0')
data_device = device if DATA_ON_DEVICE else torch.device('cpu')
print('device:', device, '| data on:', data_device)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
if has_xpu:
    torch.xpu.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(6)


# ----------------------------------------------------------------------------
# Данные: храним uint8 (1 байт на значение), во float переводим только батч
# ----------------------------------------------------------------------------
def load_raw(split, count):
    path = f'{audio_data_raw_dir}/artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_{split}.raw'
    arr = np.fromfile(path, dtype=np.uint8)  # без промежуточного bytes и без float64
    if arr.size != count * sample_len:
        raise ValueError(f'{path}: ожидалось {count * sample_len} значений, в файле {arr.size}')
    return torch.from_numpy(arr.reshape(count, sample_len))


def make_labels(per_artist):
    # данные лежат блоками по артистам; метки - индексы классов (int64), без one-hot
    return torch.arange(artist_count).repeat_interleave(per_artist)


print('loading data')
X_train = load_raw('train', count_train).to(data_device)
X_val = load_raw('validate', count_validate).to(data_device)
y_train = make_labels(count_per_artist_train).to(data_device)
y_val = make_labels(count_per_artist_validate).to(data_device)
print(f'train: {tuple(X_train.shape)} ({X_train.numel() / 2**20:.0f} MiB), '
      f'validate: {tuple(X_val.shape)} ({X_val.numel() / 2**20:.0f} MiB)')


def prep(xb):
    """uint8 [B, sample_len] -> float32 на device; нормировка прежняя: x/255 - 0.5."""
    return xb.to(device).float().div_(255.0).sub_(0.5)


def amp():
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_AMP)


# ----------------------------------------------------------------------------
# Обучение и валидация
# ----------------------------------------------------------------------------
def train_epoch(model, optimizer, criterion, epoch):
    model.train()
    n = len(X_train)
    order = torch.randperm(n).to(data_device)

    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    p_true_sum = torch.zeros((), device=device)
    seen = 0

    bar = tqdm(range(0, n, BATCH_SIZE), desc=f'epoch {epoch}', unit='batch', leave=False)
    for i, s in enumerate(bar):
        idx = order[s:s + BATCH_SIZE]
        xb = prep(X_train[idx])
        yb = y_train[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(xb)
        logits = logits.float()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        # метрики копим на устройстве, без .item() на каждом шаге (нет синхронизаций)
        with torch.no_grad():
            bs = yb.numel()
            loss_sum += loss.detach() * bs
            correct += (logits.argmax(dim=1) == yb).sum()
            p_true_sum += logits.detach().softmax(dim=1).gather(1, yb[:, None]).sum()
            seen += bs
        if i % 10 == 0:
            bar.set_postfix(loss=f'{loss_sum.item() / seen:.4f}', acc=f'{correct.item() / seen:.3f}')

    return {
        'loss': loss_sum.item() / seen,
        'acc': correct.item() / seen,
        'p_true': p_true_sum.item() / seen,
    }


@torch.no_grad()
def evaluate(model, X, y):
    """Полный проход по всей валидационной выборке, детерминированно, батчами."""
    model.eval()
    n = len(X)
    loss_sum = torch.zeros((), device=device)
    p_true_sum = torch.zeros((), device=device)
    preds = []

    for s in tqdm(range(0, n, VAL_BATCH_SIZE), desc='validation', unit='batch', leave=False):
        xb = prep(X[s:s + VAL_BATCH_SIZE])
        yb = y[s:s + VAL_BATCH_SIZE].to(device)
        with amp():
            logits = model(xb)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, yb, reduction='sum')
        p_true_sum += logits.softmax(dim=1).gather(1, yb[:, None]).sum()
        preds.append(logits.argmax(dim=1))

    preds = torch.cat(preds).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        'loss': loss_sum.item() / n,
        'acc': float((preds == y_np).mean()),
        'p_true': p_true_sum.item() / n,
        'f1': float(f1_score(y_np, preds, average='macro')),
        'preds': preds,
    }


def save_best(path, payload):
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)  # атомарно: недописанный файл не подменит хороший чекпоинт


# ----------------------------------------------------------------------------
# Основной цикл
# ----------------------------------------------------------------------------
print('preparing neural networking')
model = make_actual_nn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_GAMMA)
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
best_acc, best_loss, best_epoch = -1.0, float('inf'), -1

if RESUME and os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_acc, best_loss, best_epoch = ckpt['val_acc'], ckpt['val_loss'], ckpt['epoch']
    print(f'resumed from best checkpoint: epoch {best_epoch}, val_acc {best_acc:.4f}')

if not (RESUME and os.path.exists(log_path)):
    with open(log_path, 'w') as f:
        f.write('epoch;train_loss;train_acc;train_p_true;val_loss;val_acc;val_p_true;val_macro_f1;lr;sec\n')

print('starting training')
epochs_without_improvement = 0
epoch_bar = range(start_epoch, END_EPOCH + 1)

for epoch in epoch_bar:
    t0 = time.time()
    lr_now = optimizer.param_groups[0]['lr']

    tr = train_epoch(model, optimizer, criterion, epoch)
    scheduler.step()
    va = evaluate(model, X_val, y_val)

    # лучший = выше val accuracy, при равенстве - ниже val loss
    improved = (va['acc'] > best_acc + 1e-9) or (abs(va['acc'] - best_acc) <= 1e-9 and va['loss'] < best_loss)
    if improved:
        best_acc, best_loss, best_epoch = va['acc'], va['loss'], epoch
        epochs_without_improvement = 0
        save_best(best_path, {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': va['acc'],
            'val_loss': va['loss'],
        })
        torch.save(model.state_dict(), f'{weights_dir}/model_weights_epoch_{epoch}.pth')
    else:
        epochs_without_improvement += 1

    sec = time.time() - t0
    with open(log_path, 'a') as f:
        f.write(f"{epoch};{tr['loss']};{tr['acc']};{tr['p_true']};"
                f"{va['loss']};{va['acc']};{va['p_true']};{va['f1']};{lr_now};{sec:.1f}\n")

    tqdm.write(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
               f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1']:.4f} | "
               f"lr {lr_now:.2e} | {sec:.1f}s{'  * saved' if improved else ''}")

    if PATIENCE and epochs_without_improvement >= PATIENCE:
        tqdm.write(f'early stopping: {PATIENCE} эпох без улучшения')
        break

# ----------------------------------------------------------------------------
# Итоговая оценка лучшей модели на всей валидационной выборке
# ----------------------------------------------------------------------------
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt['model'])
final = evaluate(model, X_val, y_val)
y_np = y_val.cpu().numpy()

print(f"\nbest epoch: {ckpt['epoch']} | val loss {final['loss']:.4f} | "
      f"val acc {final['acc']:.4f} | macro-F1 {final['f1']:.4f}")
print(classification_report(y_np, final['preds'], digits=4))
print('confusion matrix (строки - истинный артист, столбцы - предсказанный):')
print(confusion_matrix(y_np, final['preds']))
