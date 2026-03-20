import numpy as np

class Node:

    def __init__(self):
        self.Name = ""
        self.Mass = 0.0 
        self.alpha = 0.0
        self.eps = 0.0
        self.Cs = 0.0
        self.PlotGroup = 0
        self.NumCells = 0
        self.Area = 0
        self.ABSFrame = np.array([0,0,0])
        self.ORBFrame = np.array([0,0,0])
        self.Qsun = 0.0
        self.Qint = 0.0
        self.Qalbedo = 0.0
        self.Qearth = 0.0
        self.Q = 0.0 
        self.Temp = 0.0

        self.TempHistory = []

        self.TempMin = 0.0
        self.TempMax = 0.0