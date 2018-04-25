import numpy as np
from os import listdir
import cv2
import os


class VidSet:
    def __init__(self, res, viz=False):
        # Video files are kept here
        self.path = 'video_files'

        # Video frame resolution
        self.res = res

        # Visualize
        self.viz = viz

        # Read video files and make numpy dataset
        self.dataset = self.make_dataset()

        print "Read dataset consisting of {} frames".format(len(self.dataset))


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
            full_f = os.path.join(self.path, f)
            cap = cv2.VideoCapture(full_f)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Raw frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Resized frame
                gray_rs = cv2.resize(gray, (self.res, self.res))

                # Add frame to list
                framelist.append(gray_rs)

                # Visualize dataset
                if self.viz:
                    cv2.imshow('frame', gray_rs)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        return np.array(framelist)


    def get_cons_sample(self, N):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N consecutive frames from the video dataset
        '''

        # Length of dataset
        d_len = len(self.dataset)

        # Get random starting point
        rnd_pt = np.random.randint(0, d_len - N + 1)

        # Get sample
        return self.dataset[rnd_pt : rnd_pt + N]


    def get_rnd_sample(self, N):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N random frames from the video dataset
        '''

        # Length of dataset
        d_len = len(self.dataset)

        # Choice arr
        choice_arr = np.random.choice(d_len, N, replace=False)

        # Chosen sample
        return self.dataset[choice_arr]



#DEBUG#
# reader = VidSet(224)
#
# dsize = len(reader.dataset)
#
# cons_sample = reader.get_cons_sample(1)
# cons_sample = reader.get_cons_sample(5)
# cons_sample = reader.get_cons_sample(dsize)
#
# rnd_sample = reader.get_rnd_sample(1)
# rnd_sample = reader.get_rnd_sample(5)
# rnd_sample = reader.get_rnd_sample(dsize)
#
# pass









