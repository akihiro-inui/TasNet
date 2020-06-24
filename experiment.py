from model.tasnet import TasNet
from model.parameters import *
import librosa


class ExperimentRunner:
    def __init__(self):
        self.__init_model()

    def __init_model(self):
        self.model = TasNet(batch_size=128, frame_length=seq_len)

    def run_one_call(self, speech_mixture):
        estimated_source = self.model.sequence(speech_mixture)

    def process_audio(self, audio_file_path: str):
        mix, _ = librosa.load(audio_file_path, sr=sr)
        return mix

    def main(self):
        # TODO: Implement sequence
        # 1. Stream audio
        # 2. Get separated source
        # 3. Calculate error
        speech_mixture = self.process_audio('data_related/data/test_audio_8k.wav')

        estimated_source = self.run_one_call(speech_mixture)


if __name__ == "__main__":
    EXR = ExperimentRunner()

    speech_mixture = EXR.process_audio('data_related/data/test_audio_8k.wav')
    a=1
