from multiprocessing import Process, Queue
import numpy as np

import pangolin
import OpenGL.GL as gl

#Global visualization usng pangolin
class Map():
    def __init__(self):
        self.frames = []  # camera frames [synonymous to camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to store current state of the map and the camera pose
        self.q = None # A queue for inter-process communication

    def create_viewer(self):
        #parallel ececution to run viewer_thread in parallel with main program

        self.q = Queue() 
        p = Process(target = self.viewer_thread)

    def viewer_thread(self, q):
        pass

