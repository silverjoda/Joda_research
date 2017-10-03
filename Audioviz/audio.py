import pyaudio
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt

class AudioDataset:
    def __init__(self, path):
        self.path = path
        self.batchsize = 1024
        self.len_seconds = 60
        self.channels = 1

        """ Init audio stream """
        self.wf = wave.open(self.path, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.channels,
            rate=self.wf.getframerate(),
            output=True
        )

        self.prepareaudiodata()


    def prepareaudiodata(self):

        frames = []
        data = self.wf.readframes(self.batchsize)
        while len(data) > 0:
            frames.append(np.fromstring(data, dtype=np.int16))
            data = self.wf.readframes(self.batchsize)

        # Convert the list of numpy-arrays into a 1D array (column-wise)
        self.np_audio_data = np.hstack(frames)

        # Rescale into 0-1 float32
        self.np_audio_data = self.np_audio_data.astype(np.float32)
        self.np_audio_data = (self.np_audio_data) / 65536


    def get_single_sample(self, samplesize):
        assert len(self.np_audio_data) > samplesize
        rnd_idx = np.random.randint(0, len(self.np_audio_data) - samplesize)
        return self.np_audio_data[rnd_idx:rnd_idx + samplesize]


    def get_batch(self, samplesize, batchsize):
        samples = []
        for i in range(batchsize):
            samples.append(self.get_single_sample(samplesize))
        return np.array(samples)


    def plot_dataset(self):
        plt.figure()
        plt.plot(range(len(self.np_audio_data)), self.np_audio_data + 0.00001)
        plt.show()

