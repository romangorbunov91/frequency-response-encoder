# Энкодер

```
python src\main.py --hypes src\config\TransformerBottleneck-model-config.json
python src\main.py --hypes src\config\UNetLike-model-config.json
python src\main.py --hypes src\config\parallelEncoder-model-config.json
python src\main.py --hypes src\config\deepEncoder-model-config.json
python src\main.py --hypes src\config\hugeKernelEncoder-model-config.json
```

## Cracks
[x] https://arxiv.org/html/2403.17725
https://arxiv.org/html/2411.04620
https://www.sciencedirect.com/science/article/abs/pii/S1474034625001429
https://www.sciencedirect.com/science/article/pii/S0950061825021129


When "Shape" matters more than Amplitude: (e.g., detecting a spike pattern regardless of how loud it is)
InstanceNorm is actually a great choice. It will make the model invariant to signal strength, focusing only on the morphology of the wave.
the model focuses on shape rather than intensity

https://arxiv.org/abs/1607.08022
https://arxiv.org/abs/2012.07436
https://arxiv.org/abs/1905.10437
https://github.com/ts-kim/RevIN

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


```
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetInstance(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose1d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv1d(64, num_classes, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.final_conv(d1)
```