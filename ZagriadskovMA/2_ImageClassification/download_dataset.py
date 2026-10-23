"""Скачивание и распаковка набора данных Intel Image Classification в каталог dataset/.

Запуск:
    python download_dataset.py                       # скачать и распаковать в dataset/
    python download_dataset.py --archive archive.zip # распаковать уже скачанный архив
    python download_dataset.py --force               # перезаписать существующий dataset/

Набор данных публичный, обычно скачивается без авторизации. Если Kaggle ответит 401/403,
используются учетные данные Kaggle API: переменные окружения KAGGLE_USERNAME и KAGGLE_KEY
или файл ~/.kaggle/kaggle.json (создается на kaggle.com: Settings -> API -> Create New Token).
"""

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"
EXPECTED_DIRS = ("seg_train/seg_train", "seg_test/seg_test", "seg_pred/seg_pred")
CHUNK_SIZE = 1 << 20

logger = logging.getLogger("download_dataset")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Скачивание набора данных Intel Image Classification")
    parser.add_argument("--url", default=DATASET_URL, help="адрес архива")
    parser.add_argument("--output", default="dataset", help="каталог для распаковки (dataset)")
    parser.add_argument("--archive", default=None,
                        help="путь к уже скачанному архиву; если файла нет, архив скачивается в этот путь")
    parser.add_argument("--keep-archive", action="store_true", help="не удалять архив после распаковки")
    parser.add_argument("--force", action="store_true", help="перезаписать существующий каталог с данными")
    return parser.parse_args(argv)


def kaggle_credentials():
    """Возвращает (username, key) из окружения или ~/.kaggle/kaggle.json, иначе None."""
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if config.is_file():
        data = json.loads(config.read_text(encoding="utf-8"))
        return data["username"], data["key"]
    return None


def open_url(url, credentials=None):
    request = urllib.request.Request(url, headers={"User-Agent": "bow-lab-downloader/1.0"})
    if credentials:
        token = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode()).decode()
        # Заголовок не передается при перенаправлении на хранилище Google Cloud,
        # у которого своя подпись в адресе
        request.add_unredirected_header("Authorization", f"Basic {token}")
    return urllib.request.urlopen(request, timeout=60)


def download(url, destination):
    """Скачивает файл по адресу url в destination с отображением прогресса."""
    try:
        response = open_url(url)
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403):
            raise
        credentials = kaggle_credentials()
        if credentials is None:
            raise RuntimeError(
                f"Kaggle вернул {e.code}. Задайте KAGGLE_USERNAME и KAGGLE_KEY или положите "
                "kaggle.json в ~/.kaggle (kaggle.com: Settings -> API -> Create New Token)") from e
        logger.info("Повторный запрос с учетными данными Kaggle API (%s)", credentials[0])
        response = open_url(url, credentials)

    total = int(response.headers.get("Content-Length", 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    logger.info("Скачивание %s (%.0f МБ) -> %s", url, total / 2**20, destination)

    done, start, last_report = 0, time.time(), 0.0
    with response, open(partial, "wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_report > 1 or (total and done == total):
                speed = done / 2**20 / max(now - start, 1e-6)
                percent = f"{100 * done / total:5.1f}%" if total else ""
                print(f"\r  {percent} {done / 2**20:7.1f} МБ  {speed:5.1f} МБ/с", end="", flush=True)
                last_report = now
    print()
    if total and done != total:
        raise RuntimeError(f"Загрузка прервана: получено {done} из {total} байт")
    partial.replace(destination)


def extract(archive, output):
    """Распаковывает zip-архив в output, не выпуская файлы за пределы каталога."""
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        logger.info("Распаковка %d файлов в %s", len(members), output)
        for i, member in enumerate(members, 1):
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Небезопасный путь в архиве: {member.filename}")
            zf.extract(member, root)
            if i % 2000 == 0 or i == len(members):
                print(f"\r  распаковано {i}/{len(members)}", end="", flush=True)
    print()


def verify(output):
    """Проверяет структуру каталога и выводит число изображений в каждой выборке."""
    for rel in EXPECTED_DIRS:
        directory = output / rel
        if not directory.is_dir():
            raise RuntimeError(f"После распаковки не найден каталог {directory}")
        count = sum(1 for p in directory.rglob("*.jpg"))
        classes = sorted(p.name for p in directory.iterdir() if p.is_dir())
        logger.info("  %-20s %6d изображений%s", rel, count, f", классы: {', '.join(classes)}" if classes else "")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    output = Path(args.output)
    try:
        if output.exists() and any(output.iterdir()):
            if not args.force:
                logger.info("Каталог %s уже существует и не пуст. Для перезаписи используйте --force", output)
                verify(output)
                return 0
            logger.info("Удаление существующего каталога %s", output)
            shutil.rmtree(output)

        archive = Path(args.archive) if args.archive else output.parent / "intel-image-classification.zip"
        downloaded = False
        if archive.is_file():
            logger.info("Используется существующий архив %s", archive)
        else:
            download(args.url, archive)
            downloaded = True

        extract(archive, output)
        verify(output)
        if downloaded and not args.keep_archive:
            archive.unlink()
            logger.info("Архив удален (используйте --keep-archive, чтобы сохранить)")
        logger.info("Готово: данные в %s", output.resolve())
    except (urllib.error.URLError, zipfile.BadZipFile, RuntimeError, OSError) as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
