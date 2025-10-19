import os
import imghdr
import mimetypes
from PIL import Image, ImageFile

# Be strict about truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Treat these as equivalent extensions
EQUIVALENT_EXTENSIONS = {
    "jpg": "jpeg",
    "jpeg": "jpg",
    "jpg": "png",
    "png": "jpg"
}

def get_image_info(path):
    """
    Analyze one file and detect any problem:
    - Corruption
    - Mislabelled extension (except jpg/jpeg equivalence)
    - Non-image disguised as image
    - Zero-byte, unreadable, or truncated files
    - Partial decode or conversion failures
    """
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    detected_format = None

    # 1️⃣ Check existence and file size
    if not os.path.exists(path):
        return False, "File missing or unreadable (OS error)", None
    try:
        size = os.path.getsize(path)
        if size == 0:
            return False, "File is 0 bytes", None
    except Exception as e:
        return False, f"Unable to read file size: {e}", None

    # 2️⃣ Header type detection
    try:
        detected_format = imghdr.what(path)
    except Exception:
        detected_format = None

    mime_type, _ = mimetypes.guess_type(path)
    if mime_type and not mime_type.startswith("image"):
        return False, f"Extension suggests non-image type ({mime_type})", detected_format

    # 3️⃣ Try opening file as image
    try:
        with Image.open(path) as img:
            pillow_format = (img.format or "").lower()

            # --- Handle jpg/jpeg equivalence ---
            def eq(a, b):
                if not a or not b:
                    return False
                if a == b:
                    return True
                if EQUIVALENT_EXTENSIONS.get(a) == b or EQUIVALENT_EXTENSIONS.get(b) == a:
                    return True
                return False

            # 4️⃣ Check mismatched extension (excluding jpg/jpeg)
            if detected_format and ext and not eq(detected_format, ext):
                return False, f"Extension mismatch: .{ext} but file header is {detected_format}", detected_format
            elif pillow_format and ext and not eq(pillow_format, ext):
                return False, f"Extension mismatch: .{ext} but Pillow detected {pillow_format}", detected_format

            # 5️⃣ Structural verification
            try:
                img.verify()
            except Exception as e:
                return False, f"PIL verify() failed ({e.__class__.__name__}: {e})", detected_format

        # 6️⃣ Fully decode and test conversion (detect truncated/partial data)
        try:
            with Image.open(path) as img:
                img.load()          # decode all data
                _ = img.convert("RGB")  # ensure full usability (e.g., grayscale conversion)
        except Exception as e:
            return False, f"Decoding failed during load/convert ({e.__class__.__name__}: {e})", detected_format

    except Exception as e:
        return False, f"Cannot open as image ({e.__class__.__name__}: {e})", detected_format

    # 7️⃣ Hidden or system/temporary file check
    if filename.startswith(".") or filename.startswith("~"):
        return False, "Hidden or temporary file name", detected_format

    # ✅ If we reached here, the file passed all tests
    return True, "OK", detected_format or "unknown"


def scan_folder_for_problems(folder_path, save_report=True, move_bad_files=False):
    """
    Scan all files in a folder for any problem.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    results = []
    bad_files = []
    bad_folder = None

    if move_bad_files:
        bad_folder = os.path.join(folder_path, "problem_files")
        os.makedirs(bad_folder, exist_ok=True)

    for fname in files:
        path = os.path.join(folder_path, fname)
        ok, message, detected = get_image_info(path)

        if ok:
            print(f"✅ {fname}: {message} [{detected}]")
        else:
            print(f"❌ {fname}: {message}")
            bad_files.append(fname)
            if move_bad_files:
                try:
                    os.replace(path, os.path.join(bad_folder, fname))
                except Exception as e:
                    print(f"   ⚠️ Could not move file: {e}")

        results.append((fname, ok, message, detected))

    # Summary
    print("\n=== SCAN SUMMARY ===")
    print(f"Total files checked: {len(files)}")
    print(f"Files with problems: {len(bad_files)}")

    if bad_files and save_report:
        report_path = os.path.join(folder_path, "problem_report.txt")
        with open(report_path, "w", encoding="utf-8") as rep:
            for name, ok, msg, det in results:
                if not ok:
                    rep.write(f"{name}: {msg}\n")
        print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    folder = r"./Data/Colored_2/"  # <-- Change this path
    scan_folder_for_problems(folder, save_report=False, move_bad_files=False)
