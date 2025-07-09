
# 🎧 NN_for_MusicPlayer

> Нейросеть для подавления шума в аудио на основе архитектуры **Dynamic NSNet2**. Используется для восстановления качества речи и музыки, с оценкой через DNSMOS.

---

## Архитектура

Модель реализует модифицированную версию [Dynamic NSNet2](https://arxiv.org/pdf/2308.16678) — рекуррентной нейросети с GRU-блоками, обученной восстанавливать амплитудный спектр звука. Используется три уровня масок (mask0, mask1, mask2) с финальной реконструкцией через ISTFT.

---

## Используемый датасет

Обучение проводилось на [DNS Challenge Dataset](https://github.com/microsoft/DNS-Challenge) от Microsoft — одном из крупнейших открытых наборов для задач шумоподавления.

---

## Структура проекта

```
NN_for_music_player/
├── model/
│   ├── dataset.py        # Подготовка спектрограмм
│   ├── train.py          # Обучение
│   ├── model.py          # DynamicNsNet2 + loss_fn
│   ├── inference.py      # Инференс модели
│   ├── eval.py           # DNSMOS оценка качества
│   ├── gen_files.py      # Шумогенерация, удаление тишины
│   └── model_v8.onnx     # ONNX-модель для оценки
├── test_model.ipynb      # Полный пример
```

---

## Быстрый старт

### Установка зависимостей

```bash
pip install torch torchaudio onnxruntime pydub numpy
```

### Подготовка аудио

Структура папок:

```
data/
├── noisy_audio/
└── clean_audio/
```

Файлы должны быть одинаковыми по названию и длительности.

### Обучение модели

```python
from model.dataset import AudioDataset
from model.train import train
from model.model import DynamicNsNet2, loss_fn
from torch.utils.data import DataLoader
import torch.optim as optim

dataset = AudioDataset("noisy_audio", "clean_audio")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = DynamicNsNet2()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train(model, loader, optimizer, loss_fn, device="cuda", num_epochs=5)
```

### Инференс

```python
from model.inference import infer
import torchaudio

denoised = infer(model, "noisy.wav", device="cuda")
torchaudio.save("denoised.wav", denoised.unsqueeze(0).cpu(), sample_rate=16000)
```

### Оценка качества

```python
from model.eval import dnsmos_score
score = dnsmos_score("denoised.wav", "model/model_v8.onnx")
print("DNSMOS Score:", score)
```

---

## 🔍 Возможности

- Dynamic NSNet2 с двумя GRU слоями и тремя выходными масками
- Обучение на спектрограммах
- Инференс с восстановлением через ISTFT
- Оценка через ONNX-модель DNSMOS
- Поддержка добавления шумов и удаления тишины (`gen_files.py`)
