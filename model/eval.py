import numpy as np
import torchaudio
import onnxruntime as ort


def load_audio(file_path, sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

def dnsmos_score(audio_path, model_path="DNSMOS.onnx"):
    ort_session = ort.InferenceSession(model_path)
    audio_data = load_audio(audio_path)

    target_size = 900 * 120
    if audio_data.size < target_size:
        padded_audio = np.pad(audio_data, (0, target_size - audio_data.size), mode='constant')
    else:
        padded_audio = audio_data[:target_size]

    padded_audio = padded_audio.reshape(1, 900, 120)

    inputs = {ort_session.get_inputs()[0].name: padded_audio}
    score = ort_session.run(None, inputs)
    return score[0]
