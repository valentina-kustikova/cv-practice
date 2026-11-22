"""Скачивание и подготовка датасета достопримечательностей ННГУ."""

import argparse
import logging
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://cloud.unn.ru/s/MPfwfnbB6HaSXLC/download"
EXPECTED_DIRS = ("trainval", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCES = {"extdataset", "nnsudataset", "ext", "nnsu"}
CHUNK_SIZE = 1 << 20

logger = logging.getLogger("download_dataset")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Скачивание набора данных достопримечательностей")
    parser.add_argument("--url", default=DATASET_URL, help="адрес архива")
    parser.add_argument("--output", default="dataset", help="каталог для распаковки (dataset)")
    parser.add_argument("--archive", default=None,
                        help="путь к уже скачанному архиву; если файла нет, архив скачивается в этот путь")
    parser.add_argument("--keep-archive", action="store_true", help="не удалять архив после распаковки")
    parser.add_argument("--force", action="store_true", help="повторно распаковать архив и обновить объединённые файлы")
    return parser.parse_args(argv)


def download(url, destination):
    """Скачивает архив; неполный файл сохраняется с расширением .part."""
    url = url.rstrip(".").rstrip("/")
    if "cloud.unn.ru/s/" in url and not url.endswith("/download"):
        url += "/download"
    request = urllib.request.Request(url, headers={"User-Agent": "bow-lab-downloader/1.0"})
    response = urllib.request.urlopen(request, timeout=60)
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


def source_roots(output):
    """Находит исходную структуру, в том числе внутри обёртки Dataset/."""
    return sorted({p.parent for p in output.rglob("*")
                   if p.is_dir() and p.name.lower() in SOURCES
                   and p.parent.name in EXPECTED_DIRS}) if output.exists() else []


def normalize(output, force=False):
    """Объединяет источники по классам; исходные файлы сохраняются."""
    for split_dir in source_roots(output):
        for source in sorted(split_dir.iterdir()):
            if not source.is_dir() or source.name.lower() not in SOURCES:
                continue
            for class_dir in sorted(source.iterdir()):
                if not class_dir.is_dir():
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    relative = path.relative_to(class_dir)
                    target = output / split_dir.name / class_dir.name / source.name / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if force or not target.exists():
                        # Жёсткая ссылка экономит место; копирование для других файловых систем.
                        if target.exists():
                            target.unlink()
                        try:
                            target.hardlink_to(path)
                        except OSError:
                            shutil.copy2(path, target)


def verify(output):
    for split in EXPECTED_DIRS:
        directory = output / split
        classes = [p for p in directory.iterdir()
                   if p.is_dir() and p.name.lower() not in SOURCES] if directory.is_dir() else []
        count = sum(1 for c in classes for p in c.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not count:
            raise RuntimeError(f"Не найдены изображения классов в {directory}")
        logger.info("%s: %d изображений, %d классов", split, count, len(classes))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    output = Path(args.output)
    try:
        downloaded = False
        archive = Path(args.archive) if args.archive else None
        if not args.force and source_roots(output):
            normalize(output)
            verify(output)
            return 0
        if not args.force and archive is None:
            try:
                verify(output)
                return 0
            except RuntimeError:
                pass
        if archive is None:
            candidates = sorted(output.glob("*.zip")) if output.exists() else []
            candidates += [output.parent / "Dataset.zip"]
            archive = next((p for p in candidates if p.is_file()), output.parent / "Dataset.zip")
        if not archive.is_file():
            download(args.url, archive)
            downloaded = True
        extract(archive, output)
        normalize(output, force=args.force)
        verify(output)
        if downloaded and not args.keep_archive:
            archive.unlink()
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
