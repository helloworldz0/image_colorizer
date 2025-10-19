import os
import re

def natural_sort_key(s):
    # Split string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def rename_files_with_numbers(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=natural_sort_key)  # ✅ natural numeric order

    for i, filename in enumerate(files, start=831):
        file_ext = os.path.splitext(filename)[1]
        new_name = f"{i}{file_ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Example usage
if __name__ == "__main__":
    folder_path = r"./Data/Colored_2/"
    rename_files_with_numbers(folder_path)
