import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


class _2ND_Orbit:

    Rearth = 0
    Fsun = 0
    Fearth = 0
    albedo = 0.3


    EclipseHistory = []

    # must be updated during orbit propagation
    Epoch = ""
    alt = 0
    ecc = 0
    INC = 0
    LTAN = 0
    RAAN = 0
    AOP = 0
    TA = 0

    @staticmethod
    def RTH_to_GCRS_Matrix(orb):
        r_dot = orb.r.value/np.linalg.norm(orb.Orbit.r.value)

        h_dot = orb.h_vec.value/np.linalg.norm(orb.Orbit.h.value)

        v_dot = np.cross(h_dot, r_dot)
        v_dot = v_dot/np.linalg.norm(v_dot)

        M_GCRS_to_RTH = np.vstack((r_dot, v_dot, h_dot))

        M_RTH_to_GCRS = M_GCRS_to_RTH.T

        return M_RTH_to_GCRS


    @staticmethod
    def RTH_to_GCRS(vector, orb):

        r_dot = orb.r.value/np.linalg.norm(orb.r.value)

        h_dot = orb.h_vec.value/np.linalg.norm(orb.h_vec.value)

        v_dot = np.cross(h_dot, r_dot)
        v_dot = v_dot/np.linalg.norm(v_dot)

        M_GCRS_to_RTH = np.vstack((r_dot, v_dot, h_dot))

        M_RTH_to_GCRS = M_GCRS_to_RTH.T

        v = M_RTH_to_GCRS @ vector
        return v

    @staticmethod
    def XYZ_to_GCRS(vector, epoch):

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
            return 1.0  
        elif separation <= (angle_earth - angle_sun):
            self.EclipseHistory.append(0.0)
            return 0.0  
        else:
            self.EclipseHistory.append((separation - (angle_earth - angle_sun)) / (2 * angle_sun))
            return (separation - (angle_earth - angle_sun)) / (2 * angle_sun)



    def create_orbit(self):

        if self.LTAN != 0:
            self.RAAN = self.ltan_to_raan(self.LTAN, self.Epoch).value

        self.Orbit = Orbit.from_classical(Earth, (self.alt+self.Rearth)/(1-self.ecc)*u.km, self.ecc*u.one, self.INC*u.rad, self.RAAN*u.rad, self.AOP*u.rad, self.TA*u.rad, self.Epoch)



    def propagate_orbit(self, Time):

        self.Orbit = self.Orbit.propagate(Time)