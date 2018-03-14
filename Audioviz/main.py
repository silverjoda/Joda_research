from audio import *
from network import AudioVizNetVer1

def main():

    # Load audio dataset
    filename='files/dont_speak-no_doubt.wav'
    audio_dataset = AudioDataset(filename)
    #audio_dataset.plot_dataset()

    # Build network


if __name__ == "__main__":
    main()