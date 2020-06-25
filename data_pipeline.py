import scipy
import librosa
import numpy as np
from model.parameters import *
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, sampling_rate: int, num_samples: int, num_frames: int, num_speakers: int):
        """
        Data loader
        :param sampling_rate: sampling rate
        :param num_samples: Number of samples in one frame
        :param num_frames: Number of frames in one audio
        :param num_speakers: Number of speakers
        """
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_speakers = num_speakers
        self.samples = {'mix': [], 's': []}
        self.sample_size = 0


    def mix_signals(self, first_signal, second_signal, SNR):
        # Generate mixture signal # TODO: At different SNR
        mixture_signal = first_signal + second_signal

        # Calculate K (number of frames from one audio) from frame length
        K = int(np.ceil(len(mixture_signal) / L))

        return mixture_signal, K

    def load_sample_from_file(self, speech_mixture_path: str):
        try:
            # Get audio
            audio, _ = librosa.load(speech_mixture_path, self.sampling_rate)
            return audio
        except FileNotFoundError:
            raise Exception(f"File does not exist {speech_mixture_path}")

    def load_all_samples(self):
        pass

    def visualize_audio(self, audio_signal):
        """
        Visualize input audio signal
        :param audio_signal:
        :return:
        """
        # Get audio duration
        duration = len(audio_signal)/self.sampling_rate
        # Create time vector
        time = np.arange(0, duration, 1/self.sampling_rate)
        plt.plot(time, audio_signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Audio wave form')
        plt.show()

    def get_sample(self, speech_mixture, first_speech, second_speech):
        sample_num = int(np.ceil(len(speech_mixture) / L))
        if sample_num < seq_len:
            sample_num = seq_len
        max_len = sample_num * L

        # Zero padding to make all signals the same length
        pad_s1 = np.concatenate([first_speech, np.zeros([max_len - len(first_speech)])])
        pad_s2 = np.concatenate([second_speech, np.zeros([max_len - len(second_speech)])])
        pad_mix = np.concatenate([speech_mixture, np.zeros([max_len - len(speech_mixture)])])

        k_ = 0
        while k_ + seq_len <= sample_num:
            begin = k_ * L
            end = (k_ + seq_len) * L
            sample_mix = pad_mix[begin:end]
            sample_s1 = pad_s1[begin:end]
            sample_s2 = pad_s2[begin:end]

            sample_mix = np.reshape(sample_mix, [seq_len, L])
            sample_s1 = np.reshape(sample_s1, [seq_len, L])
            sample_s2 = np.reshape(sample_s2, [seq_len, L])
            sample_s = np.dstack((sample_s1, sample_s2))
            sample_s = np.transpose(sample_s, (2, 0, 1))

            self.samples['mix'].append(sample_mix)
            self.samples['s'].append(sample_s)
            k_ += seq_len
        a=1


if __name__ == "__main__":
    DAL = DataLoader(8000, num_samples=L, num_frames=seq_len, num_speakers=nspk)

    speech_1 = DAL.load_sample_from_file('data/test_audio.wav')
    speech_2 = DAL.load_sample_from_file('data/test_audio.wav')
    speech_mixture = DAL.load_sample_from_file('data/test_audio_2.wav')
    # DAL.visualize_audio(speech_mixture)

    DAL.get_sample(speech_1, speech_2, speech_mixture)
