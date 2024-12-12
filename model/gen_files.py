import os
import random
from pydub import AudioSegment


def match_target_amplitude(sound, target_dBFS, offset_dB=5):
    """Увеличивает громкость звука до заданного уровня dBFS с учетом смещения."""
    change_in_dBFS = target_dBFS + offset_dB - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def get_random_noise_file(noise_folder):
    """Выбирает случайный файл шума из указанной папки."""
    noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.wav')]
    return os.path.join(noise_folder, random.choice(noise_files))


def remove_silence(sound, silence_thresh=-50.0, min_silence_len=100):
    """Удаляет тишину из аудиофайла."""
    chunks = []
    start_time = None

    for i in range(0, len(sound), min_silence_len):
        segment = sound[i:i + min_silence_len]
        if segment.dBFS > silence_thresh:
            if start_time is None:
                start_time = i
        else:
            if start_time is not None:
                chunks.append(sound[start_time:i])
                start_time = None

    if start_time is not None:
        chunks.append(sound[start_time:])

    return sum(chunks) if chunks else AudioSegment.silent(duration=0)
