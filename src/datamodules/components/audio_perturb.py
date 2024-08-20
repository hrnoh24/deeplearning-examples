import os, glob
import random
import numpy as np
import torch
import scipy
import torchaudio
from torchaudio.functional import add_noise, resample
import math

import parselmouth
from parselmouth.praat import call

from peq import parametric_equalizer

class InformationPerturbation:
    def __init__(self, noise_dataset_path, sr=16000):
        assert os.path.exists(noise_dataset_path), "Can't find noise dataset."
        
        self.noise_list = glob.glob(os.path.join(noise_dataset_path, "**/*.wav"), recursive=True)
        assert len(self.noise_list) > 0, "No noise files exist."
        
        self.sr = sr
        
    def _load_noise(self, noise_path):
        wav, sr = torchaudio.load(noise_path)
        if sr != self.sr:
            wav = resample(wav, sr, self.sr)
        return wav
    
    def _sample_ratio(self, a, b, reciprocal=.5):
        ratio = random.uniform(a, b)
        if random.random() > 0.5:
            ratio = 1 / ratio
        return ratio
    
    def fspr(self, waveform, return_tensor=True):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        
        sound = parselmouth.Sound(waveform, self.sr)
        
        # https://arxiv.org/pdf/2110.14513 참고
        fs_ratio = self._sample_ratio(1, 1.4) # U ~ [1, 1.4]
        ps_ratio = self._sample_ratio(1, 2) # U ~ [1, 2]
        pr_ratio = self._sample_ratio(1, 1.5) # U ~ [1, 1.5]
        
        # API docs : https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
        manipulation = call(sound, "Change speaker", 75, 600, fs_ratio, ps_ratio, pr_ratio, 1)
        if return_tensor:
            waveform = torch.FloatTensor(manipulation.values)
        else:
            waveform = manipulation.values
            
        return waveform
    
    def peq(self, waveform):
        waveform = parametric_equalizer(waveform, self.sr)
        return waveform
        
    def add_random_noise(self, waveform):
        noise_file = random.choice(self.noise_list)
        noise = self._load_noise(noise_file)
        
        if waveform.shape[-1] > noise.shape[-1]:
            noise = torch.functional.pad(noise, (0, waveform.shape[-1] - noise.shape[-1]))
        else:
            noise = noise[:, :waveform.shape[-1]]
        
        B = waveform.shape[0]
        ratio = torch.rand(B) * 15 # U ~ [0, 15]
        noisy_wav = add_noise(waveform, noise, ratio)
        
        return noisy_wav
    
    def apply_all(self, waveform):
        waveform = self.fspr(waveform, return_tensor=True)
        waveform = self.peq(waveform)
        waveform = self.add_random_noise(waveform)
        
        return waveform

def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)

def parametric_equalizer(wav: torch.Tensor, sr: int) -> torch.Tensor:
    cutoff_low_freq = 60.
    cutoff_high_freq = 10000.

    q_min = 2
    q_max = 5

    num_filters = 8 + 2  # 8 for peak, 2 for high/low
    key_freqs = [
        power_ratio(float(z) / (num_filters), cutoff_low_freq, cutoff_high_freq)
        for z in range(num_filters)
    ]
    Qs = [
        power_ratio(random.uniform(0, 1), q_min, q_max)
        for _ in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]

    # peak filters
    for i in range(1, 9):
        wav = apply_iir_filter(
            wav,
            ftype='peak',
            dBgain=gains[i],
            cutoff_freq=key_freqs[i],
            sample_rate=sr,
            Q=Qs[i]
        )

    # high-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='high',
        dBgain=gains[-1],
        cutoff_freq=key_freqs[-1],
        sample_rate=sr,
        Q=Qs[-1]
    )

    # low-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='low',
        dBgain=gains[0],
        cutoff_freq=key_freqs[0],
        sample_rate=sr,
        Q=Qs[0]
    )

    return wav


# implemented using the cookbook https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
def lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
    a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def apply_iir_filter(wav: torch.Tensor, ftype, dBgain, cutoff_freq, sample_rate, Q, torch_backend=True):
    if ftype == 'low':
        b0, b1, b2, a0, a1, a2 = lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'high':
        b0, b1, b2, a0, a1, a2 = highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'peak':
        b0, b1, b2, a0, a1, a2 = peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    else:
        raise NotImplementedError
    if torch_backend:
        return_wav = torchaudio.functional.biquad(wav, b0, b1, b2, a0, a1, a2)
    else:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html
        wav_numpy = wav.numpy()
        b = np.asarray([b0, b1, b2])
        a = np.asarray([a0, a1, a2])
        zi = scipy.signal.lfilter_zi(b, a) * wav_numpy[0]
        return_wav, _ = scipy.signal.lfilter(b, a, wav_numpy, zi=zi)
        return_wav = torch.from_numpy(return_wav)
    return return_wav