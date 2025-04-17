import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

raw_path = os.path.join("data", "train", "train")
audio_root = os.path.join(raw_path, "audio")
test_list_path = os.path.join(raw_path, 'testing_list.txt')
val_list_path = os.path.join(raw_path, 'validation_list.txt')
output_root = os.path.join("data", "preprocessed")

# Placeholder for preprocessing parameters
def preprocess_and_save_audio_in_tensors():
    counter = 0
    
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

            waveform, sr = torchaudio.load(full_path, num_frames=target_length, backend='soundfile')

            if sr != target_length:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            current_length = waveform.size(1)
            
            if current_length < target_length:
                # Padding too short samples
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # # Cutting too long samples
            # elif current_length > target_length:
            #     waveform = waveform[:, :target_length]

            mel = mel_transform(waveform)

            raw_out_path = Path(output_root) / "raw" / split / label / f"{filename}.pt"
            mel_out_path = Path(output_root) / "mel" / split / label / f"{filename}.pt"

            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            mel_out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(waveform, raw_out_path)
            torch.save(mel, mel_out_path)

            counter += 1

            if counter == 10000:
                print("10 000 files proccessed")
                counter = 0
                
            # print(f"Saved: {rel_path} → {split}")
            

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


class EnsembleDataset(Dataset):
    def __init__(self, path_to_raw, path_to_mel):
        self.raw_dir = Path(path_to_raw)
        self.mel_dir = Path(path_to_mel)

        self.labels = {}

        self.samples = []

        self.__prepare_file_list()

        return

    def __prepare_file_list(self):
        # check if equal categories are present
        assert [d.name for d in self.raw_dir.iterdir()] == [d.name for d in self.mel_dir.iterdir()]

        # iterate over every element
        for class_dir in self.raw_dir.iterdir():
            label = class_dir.name

            # if current element is not directory, then skip
            if not (self.raw_dir / label).is_dir() or not (self.mel_dir / label).is_dir():
                continue

            # if current category was not encountered yet, add it
            if label not in self.labels:
                self.labels[label] = len(self.labels)

            # check if both representations are present
            assert [x.name for x in (self.raw_dir / label).iterdir()] == [x.name for x in (self.mel_dir / label).iterdir()]

            for x in (self.raw_dir / label).iterdir():
                self.samples.append(((self.raw_dir / label / x.name, self.mel_dir / label / x.name), self.labels[label]))

        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        (path_to_raw, path_to_mel), label = self.samples[item]

        raw = torch.load(path_to_raw, weights_only=True)
        mel = torch.load(path_to_mel, weights_only=True)

        return (raw, mel), label