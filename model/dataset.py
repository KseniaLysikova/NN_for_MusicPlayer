import os
import torch
import torchaudio
import torchaudio.transforms as T


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, max_length=100000, n_fft=512, hop_length=256):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.clean_files = sorted(os.listdir(clean_dir))
        self.max_length = max_length
        self.stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy_waveform, _ = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)

        if self.max_length is not None:
            noisy_waveform = self._fix_length(noisy_waveform, self.max_length)
            clean_waveform = self._fix_length(clean_waveform, self.max_length)

        noisy_spectrogram = torch.log1p(self.stft_transform(noisy_waveform))
        clean_spectrogram = torch.log1p(self.stft_transform(clean_waveform))

        return noisy_spectrogram, clean_spectrogram

    def _fix_length(self, waveform, max_length):
        length = waveform.shape[-1]
        if length > max_length:
            return waveform[:, :max_length]
        elif length < max_length:
            pad_amount = max_length - length
            return torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform
