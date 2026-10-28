Инструкции по работе с подпапкой:
всё - через единый CLI — `run-filters.py`.

## Содержание

- [О проекте](#о-проекте)
- [Что понадобится](#что-понадобится)
- [Установка](#установка)
  - [Linux](#linux-ubuntu--debian--mint)
  - [macOS](#macos)
  - [Windows](#windows)
- [Проверка установки](#проверка-установки)
- [Использование `run-filters.py`](#использование-run-filterspy)
- [Список фильтров](#полный-перечень-фильтров)
- [Примеры](#примеры)
- [Пакетная обработка](#пакетная-обработка)
- [Структура проекта](#структура-проекта)
- [Частые проблемы](#частые-проблемы)

## О проекте

Структура:

- `image_filters.py` — библиотека фильтров (шум, сепия, царапины, виньетки, эффекты плёнки, бинаризация Оцу, выделение краёв и др.)
- `run-filters.py` — CLI для применения фильтров к изображению
- `imgs/` — примеры входных изображений

Изображения обрабатываются в BGR (нотация OpenCV), тип `uint8`, форма `(H, W, 3)`.

## Что понадобится

- Python **3.9 или новее** (рекомендуется 3.10+)
- pip (обычно идёт в комплекте с Python)
- git (для клонирования репозитория)
- ~50 МБ свободного места под зависимости

Зависимости Python:
- `numpy` — матричные операции
- `opencv-python` — чтение/запись изображений

## Установка

### Linux (Ubuntu / Debian / Mint)

#### Шаг 1. Python и системные зависимости

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git libgl1 libglib2.0-0

    python3-venv нужен для модуля venv

    libgl1 и libglib2.0-0 — обязательны для opencv-python, иначе import cv2 падает с libGL.so.1: cannot open shared object file

Шаг 2. Клонирование
bash

git clone <URL_репозитория>
cd <папка_репозитория>

Если репозиторий приватный и вы работаете через SSH — настройте ключи заранее. Для HTTPS git попросит логин/токен.
Шаг 3. Виртуальное окружение
bash

python3 -m venv venv
source venv/bin/activate

После source в приглашении появится (venv) — значит, python, pip теперь указывают на изолированную версию из venv/.

При следующем открытии терминала активацию нужно повторить. Деактивация — deactivate.
Шаг 4. Зависимости
bash

pip install --upgrade pip
pip install numpy opencv-python

macOS
Шаг 1. Python и git

На macOS обычно уже есть python3, но не всегда актуальной версии. Рекомендуется поставить через Homebrew:
bash

# Если Homebrew нет:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python и git:
brew install python git

Проверка: python3 --version (должен быть 3.9+).
Шаг 2. Клонирование
bash

git clone <URL_репозитория>
cd <папка_репозитория>

Шаг 3. Виртуальное окружение
bash

python3 -m venv venv
source venv/bin/activate

Синтаксис активации совпадает с Linux. На Apple Silicon (M1/M2/M3) специальных действий не нужно — numpy и opencv-python имеют готовые wheels под arm64.
Шаг 4. Зависимости
bash

pip install --upgrade pip
pip install numpy opencv-python

Windows
Шаг 1. Python и git

Скачайте установщик с https://www.python.org/downloads/ и запустите.

⚠️ Важно: на первом экране установщика обязательно отметьте галочку «Add Python to PATH» — иначе python из терминала не найдётся.

git — с https://git-scm.com/download/win.

Проверка:
powershell

python --version
git --version

Если python не находится, попробуйте py --version (лаунчер Windows). В дальнейшем вместо python можно использовать py.
Шаг 2. Клонирование

В PowerShell или CMD:
powershell

git clone <URL_репозитория>
cd <папка_репозитория>

Шаг 3. Виртуальное окружение
powershell

python -m venv venv

Активация зависит от терминала:

PowerShell:
powershell

.\venv\Scripts\Activate.ps1

Если PowerShell блокирует запуск скриптов с ошибкой running scripts is disabled, разрешите локальные скрипты (один раз):
powershell

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

CMD:
cmd

venv\Scripts\activate.bat

Git Bash / MSYS2:
bash

source venv/Scripts/activate

После активации приглашение станет вида (venv) C:\...>. Деактивация — deactivate.
Шаг 4. Зависимости
powershell

pip install --upgrade pip
pip install numpy opencv-python

Проверка установки

Из активированного venv:
bash

python -c "import cv2, numpy; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__)"

Должно вывести две строки с версиями без ошибок.
Использование run-filters.py

Скрипт применяет один или несколько фильтров к изображению и сохраняет результаты в отдельные файлы с именами <имя_входа>_<имя_фильтра>.png.
Базовый запуск
bash

python image_filters_project/run-filters.py <путь к фото>

Без флага -f применяются все зарегистрированные фильтры (~22 штуки). Результаты сохраняются в ./output/.

Только отдельные фильтры:
bash

python image_filters_project/run-filters.py <путь к фото> -f sepia -f aging

Все аргументы
Аргумент	Тип	По умолчанию	Назначение
input (позиционный)	путь	—	Входное изображение. Обязателен, если не указан --list
-f, --filter	имя (можно много раз)	[] → все	Какой фильтр применить. -f a -f b — несколько
-o, --output	путь	output	Каталог для результатов
--seed	int	42	Seed для воспроизводимости шумов/царапин
-l, --list	флаг	—	Вывести список доступных фильтров и выйти
--sepia-strength	float	1.0	Сила сепии, 0..1
--scratch-max-length	float	0.3 * min(H, W)	Максимальная длина царапины (в пикселях)

Параметры виньеток:
Аргумент	Тип	По умолчанию	Назначение
--vignette-h-semi	float	W/4	Горизонтальная полуось начального эллипса
--vignette-v-semi	float	H/4	Вертикальная полуось
--vignette-dx	float	10.0	Шаг роста по X на каждом кольце
--vignette-dy	float	10.0	Шаг роста по Y
--vignette-mode	brighten | darken	darken	Осветление или затемнение
--vignette-strength	float	1.0	Сила эффекта, 0..1
--vignette-interval-len	float	0.2	Длина интервала для случайной виньетки, 0 < L ≤ 1
Полный перечень фильтров

Актуальный список — python image_filters_project/run-filters.py --list.

Шумы и точки

    white_noise — случайные белые точки (≈2% пикселей)

    black_noise — случайные чёрные точки

    salt_pepper — «соль и перец» (белый + чёрный)

    intensity_drop — понижение яркости всего кадра вдвое

Цвет и тон

    sepia — эффект сепии

    desaturate — обесцвечивание (90% к серому)

    film_infrared — красноватый оттенок (инфракрасная плёнка)

    film_old_film — сильное затемнение (старая плёнка)

    film_overexposed — пересвет

    film_faded — выцветание (почти ч/б)

    film_vintage — тёплый жёлто-оранжевый оттенок

Геометрия и повреждения

    ellipse_mask — белая заливка вне эллипса

    scratches — царапины (ромбовидные)

    aging — комбо: сепия → царапины → случайная кольцевая виньетка

Виньетки

    vignette_rings — кольцевая виньетка (на сэтах)

    vignette_rings_masks — то же, быстрее (на масках)

    vignette_random_rings — кольцевая со случайным коэффициентом

    vignette_random_rings_masks — то же через маски

    vignette_smooth — плавная виньетка

Бинаризация и края

    otsu — бинаризация методом Оцу

    otsu_inv — то же, инвертированное

    edges — выделение краёв (Sobel + гистерезис)

    neon — неоновый эффект (края + свечение)

Примеры
bash

# Все фильтры с дефолтными параметрами
python image_filters_project/run-filters.py <путь к фото>

# Только сепия, сила 0.6
python image_filters_project/run-filters.py <путь к фото> -f sepia --sepia-strength 0.6

# Состаривание с параметрами
python image_filters_project/run-filters.py <путь к фото> -f aging --sepia-strength 0.8 --scratch-max-length 200 --vignette-h-semi 400 --vignette-v-semi 300 --vignette-strength 0.7

# Неон + плавная виньетка с осветлением по краю
python image_filters_project/run-filters.py <путь к фото> -f neon -f vignette_smooth --vignette-mode brighten --vignette-strength 0.8

# Список всех доступных фильтров
python image_filters_project/run-filters.py --list

# Свой каталог вывода
python image_filters_project/run-filters.py <путь к фото> -f edges -o <каталог>

Пакетная обработка (все изображения из папки)
Linux / macOS
bash

mkdir -p /путь/к/output

find image_filters_project/imgs -maxdepth 1 -type f \
     \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
        -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
     -print0 \
  | while IFS= read -r -d '' f; do
        echo ">>> $f"
        python image_filters_project/run-filters.py "$f" -o <путь к выходной папке>
    done

-maxdepth 1 -type f - не залазим в подпапки, перебираем все файлы, которые...
\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) - ... называются вот так.
-print0 - find выдаёт пути с '\0' в качестве разделителя
IFS= - читаем вместе с разделителями
read -r - не интерпретируем, просто сырой текст читаем
-d '' - до конца строки
f - это переменная для имени файла
echo ">>> $f" - >>> + что сейчас обрабатываем

Windows PowerShell
powershell

New-Item -ItemType Directory -Force -Path output | Out-Null

Get-ChildItem image_filters_project\imgs -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff |
  ForEach-Object {
    Write-Host ">>> $($_.FullName)"
    python image_filters_project\run-filters.py "$($_.FullName)" -o output
  }

⚠️ На Windows run-filters.py без -f прогонит все фильтры, включая медленные vignette_rings и vignette_random_rings (сэтовые версии). Для больших изображений это может занять минуты. Если слишком долго — перечислите фильтры явно через -f, исключив эти два и оставив их _masks-варианты.
Структура проекта
text

image_filters_project/
├── image_filters.py         # библиотека фильтров
├── run-filters.py           # CLI
└── imgs/                    # примеры входных изображений

Частые проблемы

python: command not found (Linux / macOS) — используйте python3. На macOS python — это старый Python 2.7, он не подойдёт.

ModuleNotFoundError: No module named 'cv2' — либо venv не активирован, либо pip install шёл в другой интерпретатор. Проверьте:
bash

which python         # Linux / macOS
where python         # Windows

Путь должен быть внутри venv/. Если нет — deactivate, потом снова source venv/bin/activate (или .\venv\Scripts\Activate.ps1 на Windows), и повторите pip install.

ImportError: libGL.so.1: cannot open shared object file (Linux) — не установлен libgl1:
bash

sudo apt install libgl1 libglib2.0-0

PowerShell не запускает Activate.ps1 — ошибка running scripts is disabled. Разрешите один раз:
powershell

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

FileNotFoundError: Не удалось прочитать: <path> — файла нет, или cv2.imread не смог открыть его (не изображение, битый файл, неподдерживаемый формат).

ValueError: Неизвестный фильтр: ... — опечатка в имени. Проверьте список: python image_filters_project/run-filters.py --list.

Эффект на выходе слабо заметен — увеличьте соответствующие параметры, например:

    --sepia-strength 1.0

    --vignette-strength 1.0

    --vignette-dx / --vignette-dy побольше — виньетка станет крупнее
