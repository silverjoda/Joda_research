from scipy.io import wavfile
import os

class Audiowriter:
    def __init__(self):
        self.path = 'audio_results'

    def writetofile(self, data, fname, rate):
        wavfile.write(os.path.join(self.path, fname), rate, data)

    def writetostream(self, data):
        raise NotImplementedError