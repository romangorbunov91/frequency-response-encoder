# Модель алгоритма идентификации параметров по частотным характеристикам

## Архитектура модели сегментации

Модель основана на архитектуре энкодер-декодер, состоящей из $N$ уровней и выполняющей преобразование ${C_{\text{in}}}$-канального тензора длины ${L_1}$ в ${C_{\text{out}}}$-канальный тензор той же длины:

$$\begin{equation*}
    \textbf{Y}: \quad \mathbb{R}^{C_{\text{in}} \times L_1} \mapsto \mathbb{R}^{C_{\text{out}} \times L_1}.
\end{equation*}$$

<p align="center" width="100%">
  <img src="./readme_img/model_block_diagram.drawio.png"
  style="background-color: white; padding: 0; width: 100%; height: auto; display: block; margin: 0 auto;" />
</p>

Количество обучаемых параметров построенной модели сегментации составляет порядка 5.3M.

Архитектура модели сегментации приведена в [base_model.py](src/models/base_model.py).

Лог обучения -- в [TransformerBottleneck-model](train_logs/TransformerBottleneck-model.json).

## Даталоудер

Создан датакласс [ZerosPolesDataset.py](src/dataloaders/ZerosPolesDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и маскам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: возврат одного примера в виде `(data_tensor, masks_tensor, freq_tensor)`.

Обучение реализовано в [train.py](src/train.py) в виде класса `ModelTrainer`.

Гиперпараметры задаются в файле [base-model-config.json](src/config/base-model-config.json).

## Работа с проектом
### 1. Скачайте файлы репозитория
### 2. Скачайте веса модели [best_TransformerBottleneck-model.pth](https://disk.yandex.ru/d/KbthwfIFpHOeFg)
### 3. Скачайте датасет [zeros-poles-masks](https://disk.yandex.ru/d/4htkLTTQqykJRQ)
### 4. Создайте окружение в директории `.venv`
```
python -m venv .venv
```
### 5. Активируйте окружение
```
.venv\Scripts\activate
```
### 6. Установите библиотеки
```
pip install -r requirements.txt
```

### Конфигурирование проекта

Общие гиперпараметры задаются в файле [config.json](src/config/config.json), включая:
- `data_dir: str` - каталог с датасетами;
- `checkpoints_dir: str` - каталог с файлами весов моделей;
- `logs_dir: str` - каталог с логами моделей;
- `score_dir: str` - каталог с картинками обучения;
- `device: str` - `cuda`, `cpu`;
- `seed: int`.

Рекомендуется работать с моделью из терминала посредством [main.py](src/main.py).

```
python -m src.main --hypes src\config\base-model-config.toml
```
или
```
python -m src.main --hypes src\config\base-model-config.toml --resume checkpoints\best_base-model_af79739e_seed1991.pth
```