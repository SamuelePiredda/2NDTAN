"""Node definition for the lumped thermal model.

Each node represents one thermal mass with its own optical properties, thermal
capacity, orientation, and bookkeeping histories used for plots and CSV export.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class Node:
    """Thermal/power state attached to one spacecraft lumped node."""


    def update_F(self, orb):
        """Estimate the Earth-view factor used for albedo and Earth IR heating."""

        radius_ratio = np.clip(orb.Rearth / np.linalg.norm(orb.Orbit.r.value), 0.0, 1.0)
        sinrho = np.square(radius_ratio)
        earth_half_angle_deg = np.rad2deg(np.arcsin(radius_ratio))

        if self.EarthAngle >= 90.0 + earth_half_angle_deg:
            self.F = 0.0
            return

        # Earth is an extended source, so flux does not drop to zero exactly at 90 deg.
        effective_angle = max(0.0, self.EarthAngle - earth_half_angle_deg)
        self.F = sinrho * np.cos(np.deg2rad(effective_angle))


    def update_angles(self, orb, epoch):
        """Refresh Sun/Earth incidence angles from the node's current normal."""
        
        if not self.Internal:
            s = get_sun(epoch).cartesian.xyz.value
            s = s/np.linalg.norm(s)

            p = np.dot(self.Normal, s)

            p = np.clip(p, -1.0, 1.0)
            p = np.rad2deg(np.acos(p))

            if p < -90 or p > 90:
                p = 90

            self.SunAngle = p
            self.SunAngleHistory.append(p)

            earth_dir = -orb.Orbit.r.value/np.linalg.norm(orb.Orbit.r.value)

            p = np.dot(self.Normal, earth_dir)
            p = np.clip(p, -1.0, 1.0)
            p = np.rad2deg(np.acos(p))

            self.EarthAngle = p
            self.EarthAngleHistory.append(p)

            self.update_F(orb)
        else:
            self.SunAngle = 90
            self.SunAngleHistory.append(90)
            self.EarthAngle = 90
            self.EarthAngleHistory.append(90)

            self.update_F(orb)





    # rotate the self.Normal vector with omega rotational speed vector and dt time step
    def rotate_Normal(self, omega, dt):
        """Rotate the current surface normal using Rodrigues' rotation formula."""
        if np.any(omega) and not self.Internal:
            omega = np.array(omega, dtype=float)
            omega_norm = np.linalg.norm(omega)

            if omega_norm == 0:
                return

            axis = omega/omega_norm
            theta = omega_norm*dt

            v_old = self.Normal/np.linalg.norm(self.Normal)
            v_new = (
                v_old*np.cos(theta)
                + np.cross(axis, v_old)*np.sin(theta)
                + axis*np.dot(axis, v_old)*(1 - np.cos(theta))
            )

            self.Normal = v_new/np.linalg.norm(v_new)
        return



    def __init__(self):
        # Static node properties read from the spreadsheet.
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
        self.Qalbedo = 0.0
        self.Qearth = 0.0
        self.Qspace = 0.0
        self.Qint = 0.0
        self.Qheater = 0.0
        self.Qint_effective = 0.0
        self.Qheater_effective = 0.0
        self.Qsolar_electric = 0.0
        self.Qbattery_loss = 0.0
        self.Q = 0.0 
        self.Temp = 0.0
        self.k = 0.0
        self.F = 0.0
        self.BatteryNode = False
        self.PowerGenerated = 0.0

        # Time histories used for plots, CSV export, and final summaries.
        self.TempHistory = []
        self.PowerGeneratedHistory = []
        self.HeaterPowerHistory = []
        self.BatteryLossHistory = []
        self.InternalPowerHistory = []

        self.TempMin = 0.0
        self.TempMax = 0.0

        self.SunAngleHistory = []
        self.EarthAngleHistory = []

        self.SunAngle = 0.0
        self.EarthAngle = 0.0

        self.Internal = False



    def updateTemp(self, temp):
        """Store the current step in history and then assign the new temperature."""

        self.TempHistory.append(self.Temp)
        self.PowerGeneratedHistory.append(self.PowerGenerated)
        self.HeaterPowerHistory.append(self.Qheater_effective)
        self.BatteryLossHistory.append(self.Qbattery_loss)
        self.InternalPowerHistory.append(self.Qint_effective)
        self.Temp = temp


    
