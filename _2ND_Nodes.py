import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class Node:


    def update_angles(self, orb, epoch):
        
        if not self.Internal:
            s = get_sun(epoch).cartesian.xyz.value
            s = s/np.linalg.norm(s)

            p = np.dot(self.Normal, s)

            p = np.clip(p, -1.0, 1.0)
            p = np.rad2deg(np.acos(p))

            self.SunAngle = p
            self.SunAngleHistory.append(p)

            r = orb.r.value/np.linalg.norm(orb.r.value)

            p = np.dot(self.Normal, r)
            p = np.clip(p, -1.0, 1.0)
            p = np.rad2deg(np.acos(p))

            self.EarthAngle = p
            self.EarthAngleHistory.append(p)
        else:
            self.SunAngle = 180
            self.SunAngleHistory.append(180)
            self.EarthAngle = 180
            self.EarthAngleHistory.append(180)





    # rotate the self.Normal vector with omega rotational speed vector and dt time step
    def rotate_Normal(self, omega, dt):
        if np.any(omega) and not self.Internal:
            dv = np.cross(omega, self.Normal)*dt
            v_new = self.Normal + dv
            v_new = v_new*(np.linalg.norm(self.Normal)/np.linalg.norm(v_new))
            self.Normal = v_new/np.linalg.norm(v_new) 
        return



    def __init__(self):
        self.Name = ""
        self.Mass = 0.0 
        self.alpha = 0.0
        self.eps = 0.0
        self.Cs = 0.0
        self.PlotGroup = 0
        self.NumCells = 0
        self.Area = 0
        self.Normal_Nadir = np.array([0,0,0])
        self.Normal_Sun = np.array([0,0,0])
        self.Normal = np.array([0,0,0])
        self.Qsun = 0.0
        self.Qint = 0.0
        self.Qalbedo = 0.0
        self.Qearth = 0.0
        self.Q = 0.0 
        self.Temp = 0.0

        self.TempHistory = []

        self.TempMin = 0.0
        self.TempMax = 0.0

        self.SunAngleHistory = []
        self.EarthAngleHistory = []

        self.SunAngle = 0.0
        self.EarthAngle = 0.0

        self.Internal = False



    def updateTemp(self, temp):

        self.TempHistory.append(self.Temp)
        self.Temp = temp


    