import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from noisereduce import reduce_noise
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

raw_path = os.path.join("data", "train", "train")
audio_root = os.path.join(raw_path, "audio")
test_list_path = os.path.join(raw_path, 'testing_list.txt')
val_list_path = os.path.join(raw_path, 'validation_list.txt')
output_root = os.path.join("data", "preprocessed")

def preprocess_and_save_audio_in_tensors(denoise = False):
    counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )
    toDB= AmplitudeToDB()

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

            waveform, sr = torchaudio.load(full_path, num_frames=target_length, backend='soundfile')

            if sr != target_length:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            current_length = waveform.size(1)
            
            if current_length < target_length:
                # Padding too short samples
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            if denoise:
                waveform = reduce_noise(waveform, sr,use_torch=True,device="cuda")
                waveform = torch.tensor(waveform)
                waveform = torch.nan_to_num(waveform,nan=0)
                output_path = os.path.join(output_root, 'denoised')
            else:
                output_path =output_path

            waveform.to(device)
            mel = mel_transform(waveform)
            mel = toDB(mel)

            raw_out_path = Path(output_path) / "raw" / split / label / f"{filename}.pt"
            mel_out_path = Path(output_path) / "mel" / split / label / f"{filename}.pt"

            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            mel_out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(waveform, raw_out_path)
            torch.save(mel, mel_out_path)


            counter += 1

            if counter == 10000:
                print("10 000 files proccessed")
                counter = 0
                
            # print(f"Saved: {rel_path} → {split}")


def preprocess_and_split_noise(
    seed=42,
    split_ratios=(0.8, 0.1, 0.1)
):
    chunk_size = 16000
    all_chunks = []
    noise_dir = os.path.join(audio_root, "_background_noise_")
    output_base_dir= os.path.join(output_root, "noise")
    
    for file in Path(noise_dir).glob("*.wav"):
        waveform, _ = torchaudio.load(file)
        total_chunks = waveform.shape[1] // chunk_size
        chunks = waveform[:, :total_chunks * chunk_size].split(chunk_size, dim=1)
        all_chunks.extend(chunks)


    torch.manual_seed(seed)
    all_chunks = [chunk for chunk in all_chunks]  
    indices = torch.randperm(len(all_chunks)).tolist()

    n_total = len(indices)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    split_sets = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train+n_val],
        'test': indices[n_train+n_val:]
    }

    for split_name, split_indices in split_sets.items():
        out_dir = os.path.join(output_base_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        for i, idx in enumerate(split_indices):
            out_path = os.path.join(out_dir, f"noise_{i:05d}.pt")

            torch.save(all_chunks[idx], out_path)

        print(f"Zapisano {len(split_indices)} fragmentów do {out_dir}")



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
        tensor = torch.load(path, weights_only=False)
        
        return tensor, label