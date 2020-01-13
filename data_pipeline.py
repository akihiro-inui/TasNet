import librosa
import numpy as np


class DataLoader:
    """
    Dataset Loader for mixture signal
    L: Number of samples in one frame
    K: Number of frames in one audio
    """
    def __init__(self):
        self.sampling_rate = 8000
        self.L = 40

    def mix_signals(self, first_signal, second_signal, SNR):
        pass

    def load_sample(self):
        # Get audio
        # mix, _ = librosa.load(mixture_path, sr=self.sampling_rate)

        # Get original source signal
        first_source_signal, _ = librosa.load(s1_path, sr=self.sampling_rate)
        second_source_signal, _ = librosa.load(s2_path, sr=self.sampling_rate)

        # Generate mixture signal # TODO: At different SNR
        mixture_signal = first_source_signal + second_source_signal

        # Calculate K (number of frames from one audio) from frame length
        K = int(np.ceil(len(mixture_signal) / self.L))



    def load_all_samples(self):
        pass