from video_dataset import VidSet
from network import VizEncoder
from audio_writer import Audiowriter
import cv2

def main():

    imres = 224
    samplerate = 44100
    n_iters = 100
    batchsize = 16

    # Make network
    network = VizEncoder("testnet", imres, samplerate)

    # Make dataset
    video_reader = VidSet(imres)

    # Train
    for i in range(n_iters):
        batch = video_reader.get_rnd_sample(batchsize)
        mse = network.train(batch)

        if i % 10 == 0:
            print "Iteration {}/{}, mse: {}".\
                format(i, n_iters, mse)

    # Test
    test_sample = video_reader.get_cons_sample(300)
    encoded_test_sample = network.encode(test_sample)
    audio_writer = Audiowriter()
    audio_writer.writetofile(encoded_test_sample,
                             "firsttest",
                             samplerate)


if __name__ == "__main__":
    main()