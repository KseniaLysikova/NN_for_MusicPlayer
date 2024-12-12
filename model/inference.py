import torch
import torchaudio
import torchaudio.transforms as T


def fix_length(waveform, max_length=100000):
    length = waveform.shape[-1]
    if length > max_length:
        return waveform[:, :max_length]
    elif length < max_length:
        pad_amount = max_length - length
        return torch.nn.functional.pad(waveform, (0, pad_amount))
    return waveform


def save_audio(waveform, filename, sample_rate=16000):
    torchaudio.save(filename, waveform.unsqueeze(0), sample_rate)


def infer(model, file_path, device="cpu", n_fft=512, hop_length=256):
    model.to(device)
    model.eval()

    waveform, sr = torchaudio.load(file_path)
    target_sr = 16000
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    waveform_fixed = fix_length(waveform)

    stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None).to(device)
    noisy_spectrogram = stft_transform(waveform_fixed.to(device))
    amplitude = torch.abs(noisy_spectrogram)
    phase = torch.angle(noisy_spectrogram)

    with torch.no_grad():
        input_amplitude = amplitude.squeeze(1)
        mask0, mask1, mask2 = model(input_amplitude)
        mask2 = mask2.squeeze(0)

    denoised_amplitude = amplitude * mask2.transpose(0, 1).unsqueeze(0)
    denoised_spectrogram = denoised_amplitude * torch.exp(1j * phase)

    istft_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)
    denoised_waveform = istft_transform(denoised_spectrogram)

    return denoised_waveform.squeeze()
