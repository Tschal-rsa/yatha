import os


def get_working_dir() -> str:
    return os.getcwd()


def get_modified_time(file: str) -> float:
    return os.stat(file).st_mtime
