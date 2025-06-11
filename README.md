# 🛠 ru-en-embedder-finetune 🚀

> Supervised Fine-Tuning русско-английского эмбеддера с использованием продвинутых техник обучения

## 📋 Описание

Этот проект представляет собой framework для fine-tuning эмбеддинг моделей на русско-английских данных с использованием supervised learning подхода. Модель обучается на триплетах (query, positive, 5 hard negatives) для улучшения качества поиска и семантического сходства.

## 🎯 Ключевые особенности

- ✅ **Multi-GPU обучение** с поддержкой Accelerate и DeepSpeed
- ✅ **Cross-Batch Memory** для увеличения эффективного размера батча
- ✅ **InfoNCE Loss** с дополнительными негативными примерами
- ✅ **Симметричное и асимметричное** обучение
- ✅ **Spherical Linear Interpolation** финальных моделей
`
`## 📁 Структура проекта

```
ru-en-embedder-finetune/
├── README.md
├── requirements.txt
├── train_one_gpu.py          # Обучение на одной GPU
├── train_accelerate.py       # Multi-GPU обучение
├── configs/
│   ├── accelerate/
│   │   └── default_config.yaml
│   ├── dataset/
│   │   ├── asymm_dataset.yaml    # Конфиг асимметричного датасета; в карточке датасета на hf есть описание
│   │   └── symm_dataset.yaml     # Конфиг симметричного датасета; в карточке датасета на hf есть описание
│   ├── model/
│   │   └── small.yaml
│   ├── train_config_asymm.yaml   # Конфиг асимметричного обучения
│   └── train_config_symm.yaml    # Конфиг симметричного обучения
├── src/
│   ├── implementations/
│   │   ├── cross_batch_memory.py      # Память негативных примеров
│   │   ├── stratified_batch_sampler.py # не используется
│   │   └── triplet_collator.py        # Формирование триплетов
│   └── utils/
│       ├── data_utils.py
│       ├── loss_utils.py
│       ├── model_utils.py
│       ├── train_accelerate_utils.py
│       ├── train_utils.py
│       └── wandb_utils.py
└── notebooks/
    ├── find-alpha-for-slerp-and-infer.ipynb
    └── semi_hard_negs_mining.ipynb
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка WandB (опционально)

```bash
export WANDB_API_KEY="ваш_ключ_wandb"
```

### 3. Обучение на одной GPU

```bash
python train_one_gpu.py --config-name train_config_symm
```

### 4. Multi-GPU обучение с Accelerate

```bash
accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  train_accelerate.py \
  --config-name  train_config_symm.yaml
```

## 🧩 Архитектура и компоненты

### TripletCollator
Формирует триплеты `(query, positive, negatives...)` с токенами-префиксами:
- `search_query` - для запросов
- `search_document` - для документов

### CrossBatchMemory  
Реализует очередь эмбеддингов предыдущих негативных примеров для увеличения эффективного размера батча и улучшения качества обучения.

### InfoNCE Loss
Стандартная InfoNCE функция потерь с дополнительными негативными примерами из памяти для более стабильного обучения.


## 📊 Результаты

### Метрики обучения
- **Симметричная модель**: [WandB Run](https://wandb.ai/alex26/ru-en-embedder-finetune-symm/runs/8cch3xzn/logs)
- **Асимметричная модель**: [WandB Run](https://wandb.ai/alex26/ru-en-embedder-finetune-symm/runs/fumrk6j6?nw=nwuseralex26)

### Финальная модель
Получена с помощью Spherical Linear Interpolation:
- Симметричная составляющая: **20%** (α = 0.2)
- Асимметричная составляющая: **80%** (α = 0.8)

### Оценка на RuMTEB

| Задача | Тип | Результат |
|--------|-----|-----------|
| CEDRClassification | MultilabelClassification | 0.373 |
| GeoreviewClassification | Classification | 0.395 |
| GeoreviewClusteringP2P | Clustering | 0.653 |
| HeadlineClassification | Classification | 0.713 |
| InappropriatenessClassification | Classification | 0.608 |
| KinopoiskClassification | Classification | 0.543 |
| RUParaPhraserSTS | STS | 0.708 |
| RuReviewsClassification | Classification | 0.581 |
| RuSTSBenchmarkSTS | STS | 0.801 |
| RuSciBenchGRNTIClassification | Classification | 0.603 |
| RuSciBenchGRNTIClusteringP2P | Clustering | 0.533 |
| RuSciBenchOECDClassification | Classification | 0.469 |
| RuSciBenchOECDClusteringP2P | Clustering | 0.458 |
| SensitiveTopicsClassification | MultilabelClassification | 0.249 |

**Общий средний результат: 0.549**

#### Результаты по типам задач:
- **Classification**: 0.559
- **Clustering**: 0.548  
- **STS (Semantic Textual Similarity)**: 0.754
- **MultilabelClassification**: 0.311


## 📚 Jupyter Notebooks

- `find-alpha-for-slerp-and-infer.ipynb` - поиск оптимального коэффициента для SLERP и инференс
- `semi_hard_negs_mining.ipynb` - майнинг полу-жёстких негативных примеров
