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

    alt = 0
    ecc = 0
    LTAN = 0
    RAAN = 0
    AOP = 0
    TA = 0

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
        raan = (sun_ra + deg_offset) % (2*np.pi * u.rad)   
        return raan


    def eclipse(self, r_sat, r_sun, r_earth=6371.0):
        """
        Checks if the satellite is in eclipse or not, accounting for penumbra.
        
        Parameters:
        r_sat : array-like [x, y, z] - satellite position vector (km)
        r_sun : array-like [x, y, z] - sun position vector (km)
        r_earth : float - Earth radius in km (default 6371.0)
        
        Returns:
        1.0 if the satellite is in full sunlight, 0.0 if it is in total eclipse (umbra),
        and a float between 0.0 and 1.0 during partial eclipse (penumbra).
        """
        r_sat = np.array(r_sat)
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
            return 1.0  
        elif separation <= (angle_earth - angle_sun):
            return 0.0  
        else:
            return (separation - (angle_earth - angle_sun)) / (2 * angle_sun)




