# Энкодер

```
python src\main.py --hypes src\config\base-model-config.json
```

When "Shape" matters more than Amplitude: (e.g., detecting a spike pattern regardless of how loud it is)
InstanceNorm is actually a great choice. It will make the model invariant to signal strength, focusing only on the morphology of the wave.
the model focuses on shape rather than intensity

https://arxiv.org/abs/1607.08022


## Структура датасета

```
project/
├── dataset/
│   └── train/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── train_masks.json
│   └── val/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── val_masks.json
│   └── test/
│       └── 0zp0lp0rp0lz0rz_000.csv
│       └── 0zp0lp0rp0lz0rz_001.csv
│       └── ...
│   └── test_masks.json
```

Каждый элемент датасета представляет собой csv-таблицу с тремя столбцами:

- Frequency (Hz),
- Gain (Real),
- Gain (Imag).

Маски агрегированы в json-файлы - по одному файлу на каждый `split`: `train`, `val`, `test`. Маски хранятся в виде целых:
- `zero_poles`: `int` - количество интеграторов,
- `left_poles`: `List[int]` - координаты полюсов левых,
- `right_poles`: `List[int]` - координаты полюсов правых,
- `left_zeros`: `List[int]` - координаты нулей левых,
- `right_zeros`: `List[int]` - координаты нулей правых.

## Даталоудер

Создан датакласс [ZerosPolesDataset.py](utils/ZerosPolesDataset.py), наследующий от `torch.utils.data.Dataset` следующие методы:
- `__init__`: инициализация путей к данным и маскам;
- `__len__`: возврат количества примеров в датасете;
- `__getitem__`: загрузка и возврат одного примера в виде `(data_tensor, masks_tensor, freq_tensor)`.

Каждый элемент представлен следующими объектами:
- `data_tensor` - тензор размера `[2, length]` - канал `real` и канал `imag`,
- `masks_tensor` - тензор размера `[4, length]` - канал полюсов левых, канал полюсов правых, канал нулей левых, канал нулей правых. Каждая маска формируется функцией [positions_to_mask](utils/ZerosPolesDataset.py) - преобразование координат нулей/полюсов в маски - массивы длины `length`, содержащие `1` в координатах нулей/полюсов и `0` в остальных,
- `freq_tensor` - тензор размера `(length,)` - частоты; в обучении автоэнкодера не требуется.

### Аугментации

В методе `_augmentations_` датакласса [ZerosPolesDataset.py](utils/ZerosPolesDataset.py) реализованы следующие виды аугментаций:
- Вырез области частот: данные + маска.
- Фазовый сдвиг в виде фиксированной задержки по времени: только данные.
- Зашумление данных: только данные.
- Умножение на константу: только данные.

Параметры агментации задаются в методе `__init__`:
- `transforms_flag`: `bool` - включение/отключение аугментаций. **Отключить: `transforms_flag=False`**.
- `crop_ratio`: `List[float]` - коэффициент вырезаемой области частот; выбирается случайным образом из диапазона `[min, max]<=1.0`. Например, 0.8 - оставить 80%. **Отключить: `crop_ratio=[1.0, 1.0]`**.
- `time_delay`: `List[float]` - сдвиг/задержка по времени; выбирается случайным образом из диапазона `[min, max]>=0.0`. **Отключить: `time_delay=[0.0, 0.0]`**.
- `noise_level`: `List[float]` - масштабирующй коэффициент шума; выбирается случайным образом из диапазона `[min, max]>=0.0`. Генерируемый шум сглаживается экспоненциально с коэффициентом `noise_filter`: `float`от 0.1 до 1.0. **Отключить: `noise_level=[0.0, 0.0]`**.
- `gain`: `List[float]` - масштабирующий коэффициент; выбирается случайным образом из диапазона `[min, max]`. Допустимо включать в диапазон положительные/отрицательные числа. **Отключить: `gain=[1.0, 1.0]`**.