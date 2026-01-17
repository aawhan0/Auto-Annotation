import cv2
import os
import shutil
from ultralytics import YOLO

# --- DYNAMIC PATH SETTING ---
# This finds the root folder regardless of where you run the script from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
AUTO_DIR = os.path.join(BASE_DIR, 'data', 'auto_labeled')
REVIEW_DIR = os.path.join(BASE_DIR, 'data', 'needs_review')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')

# --- CONFIGURATION ---
CONF_AUTO_ACCEPT = 0.85  # Accept as perfect if confidence is above this
BLUR_THRESHOLD = 100.0   # Variance of Laplacian (lower = blurrier)

# Initialize model
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    print("Please download yolov8n.pt manually and place it in the models folder.")
    exit()

model = YOLO(MODEL_PATH)

def is_blurry(img_path):
    """Checks if an image is blurry using the Laplacian method."""
    image = cv2.imread(img_path)
    if image is None: return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD

def main():
    # Ensure directories exist
    for d in [AUTO_DIR, REVIEW_DIR]:
        os.makedirs(d, exist_ok=True)

    images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    stats = {"total": 0, "auto": 0, "flagged_blur": 0, "flagged_conf": 0}
    
    print(f"🚀 Starting Audit Pipeline on {len(images)} images...")

    for filename in images:
        stats["total"] += 1
        path = os.path.join(RAW_DIR, filename)
        
        # Layer 1: Quality Audit (Blur Detection)
        if is_blurry(path):
            shutil.copy(path, os.path.join(REVIEW_DIR, f"BLUR_{filename}"))
            stats["flagged_blur"] += 1
            continue

        # Layer 2: AI Confidence Audit
        results = model(path, verbose=False)[0]
        
        # Check if model is unsure or found nothing
        is_low_confidence = False
        if len(results.boxes) == 0:
            is_low_confidence = True
        else:
            # If ANY object in the image is below our high-bar threshold
            for box in results.boxes:
                if box.conf < CONF_AUTO_ACCEPT:
                    is_low_confidence = True
                    break

        # Route the data
        if is_low_confidence:
            shutil.copy(path, os.path.join(REVIEW_DIR, f"LOWCONF_{filename}"))
            stats["flagged_conf"] += 1
        else:
            # High confidence: Save the auto-generated label and the image
            results.save_txt(os.path.join(AUTO_DIR, f"{os.path.splitext(filename)[0]}.txt"))
            shutil.copy(path, os.path.join(AUTO_DIR, filename))
            stats["auto"] += 1

    # --- FINAL SUMMARY ---
    print("\n" + "="*30)
    print("📊 AUDIT COMPLETE")
    print(f"Total Processed:  {stats['total']}")
    print(f"✅ Auto-Labeled:  {stats['auto']}")
    print(f"🚩 Flagged (Blur): {stats['flagged_blur']}")
    print(f"🚩 Flagged (Conf): {stats['flagged_conf']}")
    print("="*30)
    print(f"Check your folders: {AUTO_DIR} and {REVIEW_DIR}")

if __name__ == "__main__":
    main()