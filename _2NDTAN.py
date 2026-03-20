from _2ND_Orbit import _2ND_Orbit
from _2ND_Nodes import Node


import os
import numpy as np
import pandas as pd 
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

import tkinter as tk
from tkinter import filedialog



DATA_PROTECTION = True



SIM_VERSION = "1.0"
SIM_MAX_PARAM = 23
FILE_PATH = ""
FILE_H = ""

SIM_START_EPOCH = ""


SIM_MAX_ORBIT = 0.0
SIM_STEP_SIZE = 0.0

NUM_NODES = 0
BATTERY_NODE = 0
Nodes = []

def selectFile():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfile(title="Select a file", filetypes=[("All files", "*.*"),("XLSX files", "*.xslx"),("CSV files", "*.csv")])

    return file_path



def main():


    print("     ------------ 2NDTAN ------------")
    print("Multi-modal thermal and power simulator for cubesats")
    print("Version: " + SIM_VERSION)




    FILE_PATH = selectFile()

    if not FILE_PATH:
        print("ERROR during opening the file")
        exit(1)





    print("Extracting data...")


    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=0, header=0)
    if FILE_H.empty:
        print("ERROR during reading the NODES sheet")
        exit(1)
    FILE_H = FILE_H.fillna(0)



    NUM_NODES = 0


    # READING FIRST SHEET WITH NODES PROPERTIES
    # -------------------------------------------------------------------------------------

    for i in range(0,len(FILE_H)):

        if FILE_H.loc[i].iat[0] == 0:
            break

        NUM_NODES = NUM_NODES + 1

        nod = Node()
        nod.Name = str(FILE_H.loc[i].iat[0])

        for jj in range(0,len(Nodes)):
            if nod.Name == Nodes[jj].Name and DATA_PROTECTION:
                print("Two or more nodes have the same ID " + nod.Name)
                exit(1)

        nod.Mass = float(FILE_H.loc[i].iat[1])
        if nod.Mass < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative mass")
            exit(1)

        nod.alpha = float(FILE_H.loc[i].iat[2])
        if (nod.alpha < 0 or nod.alpha > 1) and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative or >1 alpha coefficient")
            exit(1)
        nod.eps = float(FILE_H.loc[i].iat[3])
        if (nod.eps < 0 or nod.eps > 1) and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative or >1 epsilon coefficient")
            exit(1)

        nod.Cs = float(FILE_H.loc[i].iat[4])
        if nod.Cs < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative specific heat coefficient")
            exit(1)

        nod.Area = float(FILE_H.loc[i].iat[5])/10000
        if nod.Area < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative exposed area")
            exit(1)

        nod.Qint = float(FILE_H.loc[i].iat[6])

        nod.PlotGroup = int(FILE_H.loc[i].iat[9])

        nod.NumCells = int(FILE_H.loc[i].iat[10])
        if nod.NumCells < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative number of associated cells")
            exit(1)

        if FILE_H.loc[i].iat[11] == 1:
            BATTERY_NODE = i


        # ADD NORMAL TO THE SURFACE 

        Nodes.append(nod)




    # READING SECOND SHEET WITH SIMULATION PARAMETERS
    # --------------------------------------------------------------------------------

    Orb = _2ND_Orbit()


    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=1, header=None)

    if FILE_H.empty:
        print("ERROR: The second sheet of the excel is empty")
        exit(1)

    if len(FILE_H) != SIM_MAX_PARAM:
        print("ERROR: The second sheet with all the simulation parameters is missing some")
        exit(1)


    SIM_START_EPOCH = Time(str(FILE_H.iloc[0].iat[1]), scale="utc")


    Orb.Rearth = float(FILE_H.iloc[1].iat[1])
    if (Orb.Rearth < 6000 or Orb.Rearth > 7000) and DATA_PROTECTION:
        print("ERROR: Earth radius is wrong")
        exit(1)
    
    Orb.Fsun = float(FILE_H.iloc[2].iat[1])
    if (Orb.Fsun < 0 or Orb.Fsun > 10000) and DATA_PROTECTION:
        print("ERROR: Sun Flux W/m2 is not realistic")
        exit(1)

    Orb.albedo = float(FILE_H.iloc[3].iat[1])
    if (Orb.albedo < 0 or Orb.albedo > 1) and DATA_PROTECTION:
        print("ERROR: the Earth albedo value is not realistic")
        eixt(1)

    Orb.Fearth = float(FILE_H.iloc[4].iat[1])
    if (Orb.Fearth < 0 or Orb.Fearth > 1000) and DATA_PROTECTION:
        print("ERROR: Earth Flux W/m2 is not realistic")
        exit(1)


    Orb.alt = float(FILE_H.iloc[5].iat[1])
    if (Orb.alt < 200 or Orb.alt > 10000) and DATA_PROTECTION:
        print("ERROR: The orbit altitude in km is too low or high")
        exit(1)

    Orb.ecc = float(FILE_H.iloc[6].iat[1])
    if (Orb.ecc < 0 or Orb.ecc >= 1) and DATA_PROTECTION:
        print("ERROR: the orbit eccentricity is wrong or parabolic/hyperbolic")
        exit(1)

    Orb.LTAN = float(FILE_H.iloc[7].iat[1])
    if (Orb.LTAN < 0 or Orb.LTAN > 24) and DATA_PROTECTION:
        print("ERROR: the LTAN is not valid")
        exit(1)

    Orb.RAAN = float(FILE_H.iloc[8].iat[1])
    if Orb.RAAN > 360:
        Orb.RAAN = Orb.RAAN % 360
    if Orb.RAAN < 0 and DATA_PROTECTION:
        print("ERROR: the RAAN is negative")
        exit(1)
    Orb.RAAN = Orb.RAAN*2*np.pi/360

    Orb.AOP = float(FILE_H.iloc[9].iat[1])
    if Orb.AOP > 360:
        Orb.AOP = Orb.AOP % 360
    if Orb.AOP < 0 and DATA_PROTECTION:
        print("ERROR: The orbit AOP is negative")
        exit(1)
    Orb.AOP = Orb.AOP*2*np.pi/360

    Orb.TA = float(FILE_H.iloc[10].iat[1])
    if Orb.TA > 360:
        Orb.TA = Orb.TA % 360
    if Orb.TA < 0 and DATA_PROTECTION:
        print("ERROR: The orbit TA is negative")
        exit(1)

    
    SIM_MAX_ORBIT = float(FILE_H.iloc[11].iat[1])
    if SIM_MAX_ORBIT < 0 and DATA_PROTECTION:
        print("ERROR: Orbits to simulate are negative")
        exit(1)


    SIM_STEP_SIZE = float(FILE_H.iloc[12].iat[1])
    if SIM_STEP_SIZE < 0 and DATA_PROTECTION:
        print("ERROR: Simulation step size is negative")
        exit(1)




    #TODO START TEMP
    

    print(FILE_H)



    # READING THIRD SHEET WITH EPS SIMULATION PARAMETERS
    # --------------------------------------------------------------------------------







    # READING FOURTH SHEET WITH VISIBLITY FACTOR MATRIX
    # --------------------------------------------------------------------------------






    # READING FIFTH SHEET WITH CONDUCTION AREA BEWTEEN NODES MATRIX
    # --------------------------------------------------------------------------------








    # READING SIXTH SHEET WITH DISTANCE BETWEEN NODES FOR CONDUCTION MATRIX
    # --------------------------------------------------------------------------------




    # READING SEVENTH SHEET WITH ATTITUDE MISSION TABLE
    # --------------------------------------------------------------------------------





    # READING EIGHTH SHEET WITH INTERNAL POWER NODES MISSION TABLE
    # --------------------------------------------------------------------------------




    print("Starting simulation...")





if __name__ == "__main__":
    main()