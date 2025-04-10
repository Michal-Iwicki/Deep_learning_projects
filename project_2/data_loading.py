import os
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
from torch.utils.data import Dataset

raw_path = os.path.join("data","train","train")
audio_root = os.path.join(raw_path,"audio")
test_list_path = os.path.join(raw_path,'testing_list.txt')
val_list_path = os.path.join(raw_path,'validation_list.txt')
output_root = os.path.join("data", "preprocessed")

#Propably need to run this line once 
#torchaudio.set_audio_backend("soundfile")

def preprocess_and_save_audio_in_tensors(
        # Placeholder for preprocessing parameters
):
    mel_transform = MelSpectrogram(n_mels=64)

    with open(test_list_path, 'r') as f:
        test_files = set(line.strip() for line in f if line.strip())
    with open(val_list_path, 'r') as f:
        val_files = set(line.strip() for line in f if line.strip())
        
    target_length = 16000
    for subdir, _, files in os.walk(audio_root):
        if '_background_noise_' in subdir:
            continue 

        for file in files:
            if not file.endswith(".wav"):
                continue

            full_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(full_path, audio_root).replace("\\", "/")

            if rel_path in test_files:
                split = 'test'
            elif rel_path in val_files:
                split = 'validation'
            else:
                split = 'train'

            label = rel_path.split('/')[0]
            filename = os.path.splitext(os.path.basename(file))[0]

            waveform, sr = torchaudio.load(full_path, num_frames=target_length)

            if sr != target_length:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            current_length = waveform.size(1)
            
            if current_length < target_length:
                # Padding too short samples
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # # Cutting to long samples
            # elif current_length > target_length:
            #     waveform = waveform[:, :target_length]

            mel = mel_transform(waveform)

            raw_out_path = Path(output_root) / "raw" / split / label / f"{filename}.pt"
            mel_out_path = Path(output_root) / "mel" / split / label / f"{filename}.pt"

            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            mel_out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(waveform, raw_out_path)
            torch.save(mel, mel_out_path)

            print(f"Saved: {rel_path} → {split}")

class TorchTensorFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.class_to_idx = {}
        self._prepare_file_list()

    def _prepare_file_list(self):
        for class_dir in sorted(self.root_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            for file in class_dir.glob("*.pt"):
                self.samples.append((file, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=True)
        return tensor, label