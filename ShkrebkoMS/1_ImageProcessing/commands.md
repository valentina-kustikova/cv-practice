# Команды для запуска лабораторной работы на другом компьютере

> В пути к папке проекта не должно быть русских букв — иначе `cv2.imread`
> не сможет прочитать изображение. Скачивайте проект, например, в `C:\`.

## 1. Скачать проект


git clone -b shkrebko_lab1 https://github.com/Shkrebko-Misha/cv-practice.git
cd cv-practice\ShkrebkoMS\1_ImageProcessing



## 2. Настроить окружение

Anaconda команды в **Anaconda Prompt**:

```
conda create -n cv python=3.12 -y
conda activate cv
pip install -r requirements.txt
```

только Python команды в **cmd**:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Проверка:

```
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## 3. Запустить фильтры

Сначала создать папку для результатов:

```
mkdir results
```

Все фильтры:

```
python main.py -i images/mountain.jpg -o results/resize.jpg -f resize -p 0.5
python main.py -i images/mountain.jpg -o results/gray.jpg -f gray
python main.py -i images/mountain.jpg -o results/antique.jpg -f antique -p 1
python main.py -i images/mountain.jpg -o results/fade.jpg -f fade_color -p 0.4
python main.py -i images/mountain.jpg -o results/film.jpg -f film_effect
python main.py -i images/mountain.jpg -o results/matte.jpg -f matte -p 0.6
python main.py -i images/mountain.jpg -o results/old.jpg -f old_photo -p 0.7
python main.py -i images/mountain.jpg -o results/neon.jpg -f neon -p 0.3
```

Проверка обработки ошибок:

```
python main.py -i images/nope.jpg -o results/x.jpg -f gray
python main.py -i images/mountain.jpg -o results/x.jpg -f antique -p 5
python main.py -h
```

