"""Подготовка файлов: безопасная распаковка ZIP и индексация пар JPG/PNG."""

from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import TemporaryDirectory
import zipfile
import zlib


_FOLDERS = {"train", "train_mask", "test", "test_mask"}


def _archive_inventory(
    archive: zipfile.ZipFile,
) -> list[zipfile.ZipInfo]:
    """Проверить пути и структуру ZIP до записи файлов на диск."""
    members = []
    names = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or not path.parts or path.parts[0] != "people_data"
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError(f"Unsafe archive member: {member.filename}")
        if member.is_dir():
            if len(path.parts) > 2 or (len(path.parts) == 2 and path.parts[1] not in _FOLDERS):
                raise ValueError(f"Unexpected archive directory: {member.filename}")
            continue
        if len(path.parts) != 3 or path.parts[1] not in _FOLDERS:
            raise ValueError(f"Unexpected archive member: {member.filename}")
        suffix = ".png" if path.parts[1].endswith("_mask") else ".jpg"
        if path.suffix.lower() != suffix or path.as_posix() in names:
            raise ValueError(f"Unexpected or duplicate archive member: {member.filename}")
        names.add(path.as_posix())
        members.append(member)
    if {PurePosixPath(member.filename).parts[1] for member in members} != _FOLDERS:
        raise ValueError("Archive must contain images and masks for train and test")
    return members


def _verify_cached_files(
    data_root: Path,
    members: list[zipfile.ZipInfo],
) -> None:
    """Сверить размеры и CRC распакованных файлов; ничего не перезаписывать."""
    for member in members:
        path = data_root.parent / member.filename
        if any(part.is_symlink() for part in (data_root, path.parent, path)):
            raise ValueError(f"Symlink in prepared data: {path}")
        if not path.is_file() or path.stat().st_size != member.file_size:
            raise ValueError(f"Prepared data is incomplete or changed: {path}")
        if zlib.crc32(path.read_bytes()) != member.CRC:
            raise ValueError(f"Prepared data is changed: {path}")
    expected = {data_root.parent / member.filename for member in members}
    actual = {path for path in data_root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Prepared data contains files absent from the archive")


def _extract_archive(
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    data_root: Path,
) -> None:
    """Публиковать каталог только после успешного чтения всех файлов ZIP."""
    data_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".people-data-", dir=data_root.parent) as temporary:
        for member in members:
            destination = Path(temporary) / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        # zipfile проверяет CRC при полном чтении каждого файла.
        (Path(temporary) / "people_data").rename(data_root)


def prepare_data(
    archive_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Вернуть output_dir/people_data: распаковать ZIP либо проверить готовый каталог."""
    data_root = Path(output_dir) / "people_data"
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_inventory(archive)
        if data_root.exists():
            _verify_cached_files(data_root, members)
        else:
            _extract_archive(archive, members, data_root)
    return data_root


def index_images_by_id(
    directory: str | Path,
    suffix: str,
) -> dict[str, Path]:
    """Один раз прочитать каталог; отклонить пустой набор и повторяющиеся ID."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate image IDs in {directory}")
        files[path.stem] = path
    if not files:
        raise ValueError(f"No {suffix} files in {directory}")
    return files


def index_image_mask_pairs(
    images_dir: str | Path,
    target_dir: str | Path | None = None,
) -> tuple[dict[str, Path], dict[str, Path] | None]:
    """Построить индексы JPG и PNG; при наличии масок потребовать совпадения ID."""
    images = index_images_by_id(images_dir, ".jpg")
    masks = index_images_by_id(target_dir, ".png") if target_dir is not None else None
    if masks is not None and images.keys() != masks.keys():
        raise ValueError(f"Image/mask IDs differ in {images_dir} and {target_dir}")
    return images, masks
