import os


def is_valid_file(file_path: str) -> bool:
    """
    Confirm the file path is valid
    :param   file_path: file path
    :return  True if it is valid file and False if it is invalid file path
    """
    return os.path.isfile(file_path)
