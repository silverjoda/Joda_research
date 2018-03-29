from video_dataset import VidSet
from network import VizEncoder

def main():

    # Make network
    network = VizEncoder("testnet", 224, 44100)

    # DEBUGGING: Encoder seems to be working
    #            Decoder: In progress

if __name__ == "__main__":
    main()