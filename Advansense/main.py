from video_dataset import VidSet
from network import VizEncoder

def main():

    imres = 224
    samplerate = 44100
    n_iters = 100
    batchsize = 16

    # Make network
    network = VizEncoder("testnet", imres, samplerate)

    # Make dataset
    reader = VidSet(imres)

    # Train

    for i in range(n_iters):
        batch = reader.get_rnd_sample(batchsize)



if __name__ == "__main__":
    main()