# src/data/preprocessing.py

import os
from PIL import Image

def preprocess_and_save_images(data_dir, processed_dir, target_size=(224, 224)):
    # Create processed folder structure
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    for split in splits:
        for class_name in classes:
            folder = os.path.join(processed_dir, split, class_name)
            os.makedirs(folder, exist_ok=True)
    
    processed_count = 0
    
    for split in splits:
        for class_name in classes:
            source_folder = os.path.join(data_dir, split, class_name)
            target_folder = os.path.join(processed_dir, split, class_name)
            
            if os.path.exists(source_folder):
                files = os.listdir(source_folder)
                
                for file in files:
                    try:
                        img_path = os.path.join(source_folder, file)
                        img = Image.open(img_path)
                        
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        
                        save_path = os.path.join(target_folder, file)
                        img_resized.save(save_path, 'JPEG', quality=95)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
    
    print(f"✅ Preprocessing complete! Processed {processed_count} images")
