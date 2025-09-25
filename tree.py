import os
from pathlib import Path

def print_tree(startpath: str | Path, ignore: tuple[str, ...] = ("vani_env", "__pycache__")) -> None:
    """
    Recursively prints the directory structure starting from `startpath`.
    
    Args:
        startpath (str | Path): Root directory to start printing from.
        ignore (tuple[str, ...]): Directory names to ignore.
    """
    startpath = Path(startpath).resolve()

    for root, dirs, files in os.walk(startpath):
        # Filter ignored directories
        dirs[:] = [d for d in dirs if d not in ignore]

        # Compute depth level relative to startpath
        rel_path = Path(root).relative_to(startpath)
        level = len(rel_path.parts)

        # Indentation logic
        indent = "│   " * (level - 1) + ("├── " if level > 0 else "")

        # Print folder name
        folder_name = root if root == str(startpath) else os.path.basename(root)
        print(f"{indent}{folder_name}/")

        # Print files inside the folder
        for i, file in enumerate(files):
            file_indent = "│   " * level
            connector = "└── " if i == len(files) - 1 else "├── "
            print(f"{file_indent}{connector}{file}")


if __name__ == "__main__":
    print_tree(".", ignore=("vani_env", "__pycache__", ".git", ".idea"))