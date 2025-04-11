import os
import shutil

def move_audio_files(train_dir='data/train', test_dir='data/test', val_dir='data/validation', test_list_path='testing_list.txt', val_list_path='validation_list.txt'):
    with open(test_list_path, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    with open(val_list_path, 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]

    def move_file(file_name, target_folder):
        src_path = os.path.join(train_dir, file_name)
        dst_path = os.path.join(target_folder, file_name)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} → {dst_path}")
        else:
            print(f"File not found: {src_path}")

    for file in test_files:
        move_file(file, test_dir)

    for file in val_files:
        move_file(file, val_dir)
