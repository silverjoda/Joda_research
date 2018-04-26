import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

class Audiowriter:
    def __init__(self):
        self.path = 'audio_results'

    def writetofile(self, data, fname, rate):
        data = data / np.max(data)
        data = data * 2 - np.max(data)
        print data.shape
        plt.plot(data[:44111])
        plt.show()
        print "Writing data type: {}, min: {}, max: {}".format(data.dtype, np.min(data), np.max(data))
        wavfile.write(os.path.join(self.path, fname), rate, data)

    def writetostream(self, data):
        raise NotImplementedError


#data = np.random.randn(20000)
#wavfile.write("shitfile.wav", 44100,data)