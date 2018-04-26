from video_dataset import VidSet
from network import VizEncoder
from audio_writer import Audiowriter
import numpy as np
import cv2

def main():

    LOAD = False

    imres = 224
    samplerate = 44100
    n_iters = 2000
    batchsize = 16

    # Make network
    network = VizEncoder("testnet", imres, samplerate)

    # Make dataset
    video_reader = VidSet(imres)

    if LOAD:
        # Load weights
        network.restore_weights()
    else:
        # Train
        for i in range(n_iters):
            batch = video_reader.get_rnd_sample(batchsize)
            mse = network.train(batch)

            if i % 10 == 0:
                print "Iteration {}/{}, mse: {}".\
                    format(i, n_iters, mse)

        # Save weights
        network.save_weights()

    # Test
    test_sample = video_reader.get_cons_sample(300)
    encoded_test_sample = network.encode(test_sample)
    concatenated_result = np.reshape(encoded_test_sample, [-1])
    audio_writer = Audiowriter()
    audio_writer.writetofile(concatenated_result,
                             "firsttest.wav",
                             samplerate)




if __name__ == "__main__":
    main()