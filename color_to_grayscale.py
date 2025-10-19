import cv2
import os
import glob

# === Configuration ===
source_folder = r'./Data/Colored_2/'
destination_folder = r'./Data/Grayscale_2/'

# === Setup ===
success = 0
failure = 0

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Collect all supported image extensions
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')

image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(source_folder, ext)))

# Normalize paths and remove duplicates (case-insensitive on Windows)
image_paths = list({os.path.normcase(os.path.normpath(p)) for p in image_paths})

print(f"Found {len(image_paths)} candidate images.\n")

# === Process each image ===
for img_path in sorted(image_paths):
    # Skip non-files (in case folders slip in)
    if not os.path.isfile(img_path):
        print(f"Skipping non-file: {img_path}")
        continue

    # Extract just the filename
    filename = os.path.basename(img_path)

    # Try loading the image
    color_img = cv2.imread(img_path)

    # Check if load failed (corrupted, mislabeled, etc.)
    if color_img is None:
        print(f"❌ Failed to load image: {filename}")
        failure += 1
        continue

    try:
        # Convert to grayscale
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # Save to destination folder
        output_path = os.path.join(destination_folder, filename)
        cv2.imwrite(output_path, gray_img)

        print(f"✅ Converted and saved: {output_path}")
        success += 1

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")
        failure += 1

print("\n=== SUMMARY ===")
print(f"Total processed: {len(image_paths)}")
print(f"Successfully converted: {success}")
print(f"Failed or unreadable: {failure}")
