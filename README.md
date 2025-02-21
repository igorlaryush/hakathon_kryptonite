# Хакатон Криптонита

## Общее описание задачи

Целью хакатона является построения модели для построения эмбеддингов изображений лиц устойчивой к атакам, которые используют синтетические данные.


## Данные

Поместить артефакты в папку проекта, а именно:
- test.csv
- train.csv
- val.csv
- data/

## Описание данных

Данные - реальные и синтетически сгенерированные изображения лиц расположены в папке `data`.

```
for_codenrock
├── test_public  # тестовые данные
│   ├── 00000000    # pair_id - ID сравниваемой пары сообщений
│   │   ├── 0.jpg
│   │   └── 1.jpg
│   ├── 00000001
│   │   ├── 0.jpg
│   │   └── 1.jpg
...
└── train   # данные для обучения модели
    ├── 00000000    # label - ID сравниваемой пары сообщений
    │   ├── 0.jpg
    │   ├── 0.jpg
    │   ├── 0.jpg
    │  ...
    │   └── k_0.jpg
    ├── 00000001
    │   ├── 0.jpg
    │   ├── 0.jpg
    │   ├── 0.jpg
    │  ...
    │   └── k_1.jpg
...
```

train.csv, val.csv, test.csv - таблицы с разметкой тренировочных, валидационных и тестовых данных соответственно.

Содержат следующие поля: 

```
label - метка объекта, уникальный идентификатор лица (у разных людей разные id). В случае test.csv метка сравнения, нужна для формирования submission.csv.
path - относительный путь к изображению
split - метка разбиения (train, validation или test)
is_query - флаг, обозначающий является ли изображение запросом
is_gallery - флаг, обозначающий находится ли изображение в галерее
```

### Структура submission.csv

submission.csv - это файл формата ".csv" с разделителем ",", который содержит две колонки:

- `pair_id` - название директории, в которой расположена пара сравниваемых изображений
- `similarity` - характеристика похожести двух изображений в виде произвольного вещественного числа


Пример файла (в `./data/sample_submission.csv`):
```
pair_id,similarity
00000000,1.0
00000001,0.9999271
00000002,0.99991727
```


## Бейзлайн решение

### Окружение
Версия Python: `Python 3.10.12`.
  
Создание окружения:

```bash
VENV_DIR="../../venvs/hakathon_kryptonite"
python3 -m virtualenv $VENV_DIR
source $VENV_DIR/bin/activate

pip install -r requirements.txt
```


Минимальные требования:

1 GB VRAM - inference

6 GB VRAM - train

### Запуск скриптов
1. Обучаем модель.
**Модель обучается локально!**

```bash
python train.py
```

2. Формируем `submission.csv` с использованием построенной модели.
**Сабмит формируется локально!**

```
python make_submission.py 
```


## Целевая метрика


Error equal rate - https://www.innovatrics.com/glossary/equal-error-rate-eer/.

Код для вычисления метрики:
```py
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)

    # заменяем np.inf на max + eps
    eps = 1e-3
    threshold[0] = max(threshold[1:]) + eps

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fnr[eer_index]
    return eer
```


## Пример загрузки submission на платформу

### 1. Клонируйте проект:

```bash
git clone git@git.codenrock.com:kryptonite-ml-challenge-1347/template_6468.git
```

### 2. Загрузите данные и поместите в папку data:

Данные для обучения: [ссылка для скачивания](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/train.zip)
Данные для теста: [ссылка для скачивания](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip)

### 3. Обучите модель

```bash
python train.py
```

### 4. Создайте submission.csv

```bash
python make_submission.py 
```

### 5. Загрузите решение на площадку

```bash
git add .
git commit -m 'submission'
git push origin master
```