# ============================================================================
# src/utils/file_utils.py
# ============================================================================
"""
File utilities for medical ingestion engine.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import hashlib
import json


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path to directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes
    """
    return file_path.stat().st_size


def is_pdf(file_path: Path) -> bool:
    """
    Check if file is a PDF.

    Args:
        file_path: Path to file

    Returns:
        True if PDF
    """
    if not file_path.exists():
        return False

    # Check extension
    if file_path.suffix.lower() != '.pdf':
        return False

    # Check magic bytes
    with open(file_path, 'rb') as f:
        header = f.read(4)
        return header == b'%PDF'


def list_pdfs(directory: Path) -> List[Path]:
    """
    List all PDF files in directory.

    Args:
        directory: Directory to search

    Returns:
        List of PDF file paths
    """
    if not directory.exists():
        return []

    return [f for f in directory.rglob('*.pdf') if f.is_file()]


def copy_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
    """
    Copy file to destination.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to copied file
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {destination}")

    ensure_directory(destination.parent)
    shutil.copy2(source, destination)

    return destination


def move_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
    """
    Move file to destination.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to moved file
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {destination}")

    ensure_directory(destination.parent)
    shutil.move(str(source), str(destination))

    return destination


def delete_file(file_path: Path, missing_ok: bool = True) -> None:
    """
    Delete file.

    Args:
        file_path: Path to file
        missing_ok: Don't raise error if file doesn't exist
    """
    if not file_path.exists() and not missing_ok:
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.exists():
        file_path.unlink()


def read_json(file_path: Path) -> dict:
    """
    Read JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: dict, file_path: Path, indent: int = 2) -> None:
    """
    Write JSON file.

    Args:
        data: Data to write
        file_path: Path to JSON file
        indent: Indentation level
    """
    ensure_directory(file_path.parent)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Get relative path from base.

    Args:
        path: Absolute path
        base: Base directory

    Returns:
        Relative path
    """
    return path.relative_to(base)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


def get_unique_filename(directory: Path, base_name: str, extension: str = '') -> Path:
    """
    Get unique filename by appending number if file exists.

    Args:
        directory: Directory for file
        base_name: Base filename
        extension: File extension (with or without dot)

    Returns:
        Unique file path
    """
    if not extension.startswith('.') and extension:
        extension = '.' + extension

    path = directory / f"{base_name}{extension}"

    if not path.exists():
        return path

    counter = 1
    while True:
        path = directory / f"{base_name}_{counter}{extension}"
        if not path.exists():
            return path
        counter += 1


def clean_directory(directory: Path, pattern: str = '*', keep_count: Optional[int] = None) -> int:
    """
    Clean files from directory matching pattern.

    Args:
        directory: Directory to clean
        pattern: Glob pattern for files to delete
        keep_count: Optional number of newest files to keep

    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0

    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if keep_count is not None:
        files_to_delete = files[keep_count:]
    else:
        files_to_delete = files

    for file_path in files_to_delete:
        if file_path.is_file():
            file_path.unlink()

    return len(files_to_delete)


def get_directory_size(directory: Path) -> int:
    """
    Get total size of directory in bytes.

    Args:
        directory: Directory path

    Returns:
        Total size in bytes
    """
    total_size = 0

    for path in directory.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size

    return total_size
