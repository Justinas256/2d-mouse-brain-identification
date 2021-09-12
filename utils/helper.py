import os


def create_folder_if_not_exists(file_path: str):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        print(f"Creating folder {dir_path}")
        os.makedirs(dir_path)
