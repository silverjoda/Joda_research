import numpy as np
from os import listdir
import cv2


class VidSet:
    def __init__(self, res):
        # Video files are kept here
        self.path = 'video_files'

        # Video frame resolution
        self.res = res

        # Read video files and make numpy dataset
        self.dataset = self.make_dataset()

    def make_dataset(self):
        '''

        Returns
        -------
        Numpy array dataset of video frames from all video files

        '''

        # Frames will be stored here
        framelist = []

        # Check all available files
        files = listdir(self.path)

        for f in files:
            cap = cv2.VideoCapture(f)

            while (cap.isOpened()):
                ret, frame = cap.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Add frame to list
                framelist.append(gray)

                cv2.imshow('frame', gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        return np.array(framelist)



    def get_cons_sample(self, N, res):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N consecutive frames from the video dataset
        '''
        pass

    def get_rnd_sample(self, N, res):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N random frames from the video dataset
        '''
        pass
