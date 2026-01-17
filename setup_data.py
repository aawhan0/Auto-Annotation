import os
import shutil

# Change this to where you unzipped the Roboflow folder
SOURCE_IMG_DIR = r'C:\Users\vyasa\Downloads\Construction Site Safety.v30-raw-images_latestversion.yolov8\train\images'
DEST_RAW_DIR = os.path.join(os.getcwd(), 'data', 'raw')

def setup():
    if not os.path.exists(DEST_RAW_DIR):
        os.makedirs(DEST_RAW_DIR)
        
    files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"Moving {len(files)} images to data/raw...")
    for f in files:
        shutil.copy(os.path.join(SOURCE_IMG_DIR, f), os.path.join(DEST_RAW_DIR, f))
    print("✅ Setup complete. You can now run the audit pipeline.")

if __name__ == "__main__":
    setup()