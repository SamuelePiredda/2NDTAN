"""Orbital geometry utilities used by the thermal and power simulator.

This module stores the current orbit state and exposes helper transforms that
convert between mission-friendly frames (RTH, Sun-aligned, inertial XYZ) and
the inertial frame used by the rest of the simulation.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


class _2ND_Orbit:
    """Container for the spacecraft orbit and environment-dependent quantities."""

    def __init__(self):
        # Global environment values used by the thermal/power models.
        self.Rearth = 0
        self.Fsun = 0
        self.Fearth = 0
        self.albedo = 0.3
        self.EclipseHistory = []
        self.Eclipse = 0

        # must be updated during orbit propagation
        self.Epoch = ""
        self.alt = 0
        self.ecc = 0
        self.INC = 0
        self.LTAN = None
        self.RAAN = 0
        self.AOP = 0
        self.TA = 0

    @staticmethod
    def RTH_to_GCRS_Matrix(orb):
        """Build the rotation matrix from the local RTH frame to inertial GCRS."""
        r_dot = orb.r.value/np.linalg.norm(orb.r.value)

        h_dot = orb.h_vec.value/np.linalg.norm(orb.h_vec.value)

        v_dot = np.cross(h_dot, r_dot)
        v_dot = v_dot/np.linalg.norm(v_dot)

        M_GCRS_to_RTH = np.vstack((r_dot, v_dot, h_dot))

        M_RTH_to_GCRS = M_GCRS_to_RTH.T

        return M_RTH_to_GCRS


    @staticmethod
    def RTH_to_GCRS(vector, orb):
        """Rotate a vector written in the local RTH frame into inertial GCRS."""

        r_dot = orb.r.value/np.linalg.norm(orb.r.value)

        h_dot = orb.h_vec.value/np.linalg.norm(orb.h_vec.value)

        v_dot = np.cross(h_dot, r_dot)
        v_dot = v_dot/np.linalg.norm(v_dot)

        M_GCRS_to_RTH = np.vstack((r_dot, v_dot, h_dot))

        M_RTH_to_GCRS = M_GCRS_to_RTH.T

        v = M_RTH_to_GCRS @ vector
        return v

    @staticmethod
    def SUNXYZ_to_GCRS(vector, epoch):
        """Rotate a vector from a Sun-aligned frame into inertial GCRS."""

        S = get_sun(epoch).cartesian.xyz.value

        X_dot = S/np.linalg.norm(S)

        Zearth = np.array([0,0,1])

        Y_dot = np.cross(Zearth, X_dot)
        Y_dot = Y_dot/np.linalg.norm(Y_dot)

        Z_dot = np.cross(X_dot, Y_dot)
        Z_dot = Z_dot/np.linalg.norm(Z_dot)

        M_GCRS_to_XYZ = np.vstack((X_dot, Y_dot, Z_dot))
        M_XYZ_to_GCRS = M_GCRS_to_XYZ.T
        return M_XYZ_to_GCRS @ vector

    @staticmethod
    def INERTIALXYZ_to_GCRS(vector):
        """Pass-through helper for vectors already expressed in inertial XYZ."""
        return np.array(vector, dtype=float)

    @staticmethod
    def get_nadir_rotation_vector(orb):
        """Angular-rate vector that keeps the spacecraft nadir-pointing."""
        r_norm = np.linalg.norm(orb.r.value)
        return orb.h_vec.value/(r_norm*r_norm)




    def ltan_to_raan(self,ltan_hours, epoch):
        """
        Converts LTAN to RAAN for a specific epoch.
        
        Parameters:
        ltan_hours : float - LTAN expressed in decimal hours (es. 10.5 for 10:30)
        epoch    : astropy.time.Time - reference epoch
        
        Returns:
        raan : astropy.units.Quantity - RAAN in degree (between 0 and 360)
        """
        sun_pos = get_sun(epoch)
        sun_ra = sun_pos.ra
        deg_offset = (ltan_hours - 12.0) * 15.0
        deg_offset = deg_offset*u.deg
        raan = (sun_ra + deg_offset) % (2*np.pi * u.rad)   
        return raan


    def eclipse(self):
        """
        Checks if the satellite is in eclipse or not, accounting for penumbra.
        
        Parameters:        
        Returns:
        1.0 if the satellite is in full sunlight, 0.0 if it is in total eclipse (umbra),
        and a float between 0.0 and 1.0 during partial eclipse (penumbra).
        """
        r_sat = self.Orbit.r.to(u.km).value
        r_sun = get_sun(self.Epoch).cartesian.xyz.to(u.km).value

        r_sun = np.array(r_sun)
        r_sun_radius = 696340.0 

        v_earth = -r_sat
        v_sun = r_sun - r_sat
        
        dist_earth = np.linalg.norm(v_earth)
        dist_sun = np.linalg.norm(v_sun)

        angle_earth = np.arcsin(np.clip(self.Rearth / dist_earth, 0.0, 1.0))
        angle_sun = np.arcsin(np.clip(r_sun_radius / dist_sun, 0.0, 1.0))

        v_earth_unit = v_earth / dist_earth
        v_sun_unit = v_sun / dist_sun
        
        dot_prod = np.dot(v_earth_unit, v_sun_unit)
        separation = np.arccos(np.clip(dot_prod, -1.0, 1.0))

        if separation >= (angle_earth + angle_sun):
            self.EclipseHistory.append(1.0)
            self.Eclipse = 1.0
            return 1.0  
        elif separation <= (angle_earth - angle_sun):
            self.EclipseHistory.append(0.0)
            self.Eclipse = 0.0
            return 0.0  
        else:
            self.EclipseHistory.append((separation - (angle_earth - angle_sun)) / (2 * angle_sun))
            self.Eclipse = (separation - (angle_earth - angle_sun)) / (2 * angle_sun)
            return (separation - (angle_earth - angle_sun)) / (2 * angle_sun)



    def create_orbit(self):
        """Create the poliastro orbit object from the spreadsheet parameters."""

        if self.LTAN is not None:
            self.RAAN = self.ltan_to_raan(self.LTAN, self.Epoch).to(u.rad).value

        # The input altitude is defined as the pericenter altitude above Earth.
        self.Orbit = Orbit.from_classical(Earth, (self.alt+self.Rearth)/(1-self.ecc)*u.km, self.ecc*u.one, self.INC*u.rad, self.RAAN*u.rad, self.AOP*u.rad, self.TA*u.rad, self.Epoch)



    def propagate_orbit(self, delta_t):
        """Advance the stored orbit state by the given propagation time."""

        self.Orbit = self.Orbit.propagate(delta_t)
