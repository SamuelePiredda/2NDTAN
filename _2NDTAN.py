from _2ND_Orbit import _2ND_Orbit
from _2ND_Nodes import Node

import math
import os
import random
import numpy as np
import pandas as pd 
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog



def DEBUG_printNodes(node_array):
    """
    Iterates through an array of node objects and prints their parameters 
    in a cleanly formatted way for debugging.
    """
    print(f"=== Debugging {len(node_array)} Elements ===")
    
    for i, obj in enumerate(node_array):
        print(f"\n--- Element [{i}]: {obj.Name if obj.Name else 'Unnamed'} ---")
        
        # Identity & Geometry
        print(f"  Geometry   | Mass: {obj.Mass} | Area: {obj.Area} | NumCells: {obj.NumCells}")
        print(f"  Properties | alpha: {obj.alpha} | eps: {obj.eps} | Cs: {obj.Cs} | PlotGroup: {obj.PlotGroup}")
        print(f"  Status     | Internal: {obj.Internal}")
        
        # Vectors (Assuming NumPy arrays)
        print("  Vectors    |")
        print(f"    Nadir    : {obj.Normal_Nadir}")
        print(f"    Sun      : {obj.Normal_Sun}")
        print(f"    Normal   : {obj.Normal}")
        
        # Angles
        print(f"  Angles     | Sun: {obj.SunAngle:.4f}° | Earth: {obj.EarthAngle:.4f}°")
        
        # Heat Fluxes (Q)
        print("  Heat Flux  |")
        print(f"    Qsun     : {obj.Qsun}")
        print(f"    Qalbedo  : {obj.Qalbedo}")
        print(f"    Qearth   : {obj.Qearth}")
        print(f"    Qint     : {obj.Qint}")
        print(f"    Q (Total): {obj.Q}")
        
        # Temperatures
        print("  Temperat.  |")
        print(f"    Current  : {obj.Temp}")
        print(f"    Min/Max  : {obj.TempMin} / {obj.TempMax}")
        
        # TempHistory (printing length to avoid terminal flooding)
        history_len = len(obj.TempHistory)
        print(f"    History  : [{history_len} data points recorded]")
        
    print("\n=== End of Debug Info ===")





def set_Rotational_Vector(attitude, custom, random_coeff, orb):

    if attitude == "S":
        return np.array([0,0,0])
    elif attitude == "N":
        return _2ND_Orbit.RTH_to_GCRS(np.array([0, 0, 2*np.pi/orb.period.to("s").value]), orb)
    elif attitude == "R":
        X = (random.random() - 0.5)*2*random_coeff
        Y = (random.random() - 0.5)*2*random_coeff
        Z = (random.random() - 0.5)*2*random_coeff
        return np.array([X,Y,Z])
    elif attitude == "C":
        return custom
    # INTERTIAL POINTING
    elif attitude == "F":
        return np.array([0,0,0])
    else:
        print("ERROR: error during setting the rotataional vector with attitude " + str(attitude))
        exit(1)




def set_Nodes_Nadir(Nodes, attitude, orb, epoch):
    for nod in Nodes:
        nod.Normal = setup_norm_vector(nod.Normal_Nadir, attitude, orb, epoch)

def set_Nodes_Sun(Nodes, attitude, orb, epoch):
    for nod in Nodes:
        nod.Normal = setup_norm_vector(nod.Normal_Sun, attitude, orb, epoch)


def setup_norm_vector(text, attitude, orb, epoch):

    text = text.upper()

    if attitude == "N":
        if text == "R" or text == "+R":
            return _2ND_Orbit.RTH_to_GCRS([1,0,0], orb)
        elif text == "-R":
            return _2ND_Orbit.RTH_to_GCRS([-1,0,0], orb)
        elif text == "T" or text == "+T":
            return _2ND_Orbit.RTH_to_GCRS([0,1,0], orb)
        elif text == "-T":
            return _2ND_Orbit.RTH_to_GCRS([0,-1,0], orb)
        elif text == "H" or text == "+H":
            return _2ND_Orbit.RTH_to_GCRS([0,0,1], orb)
        elif text == "-H":
            return _2ND_Orbit.RTH_to_GCRS([0,0,-1], orb)
        elif text == "I":
            return 0
        else:
            try:
                tmp = text.split(',')
                R = float(tmp[0])
                T = float(tmp[1])
                H = float(tmp[2])
                tmp = np.array([R,T,H])
                tmp = tmp/np.linalg.norm(tmp)
                return _2ND_Orbit.RTH_to_GCRS(tmp, orb)
            except Exception as e:
                print("ERROR: During parsing the norm of one node the string is not correct " + str(text) + " '"+str(attitude)+"'")
                exit(1)

    else:
        if text == "X" or text == "+X":
            return _2ND_Orbit.XYZ_to_GCRS([1,0,0], epoch)
        elif text == "-X":
            return _2ND_Orbit.XYZ_to_GCRS([-1,0,0], epoch)
        elif text == "Y" or text == "+Y":
            return _2ND_Orbit.XYZ_to_GCRS([0,1,0], epoch)
        elif text == "-Y":
            return _2ND_Orbit.XYZ_to_GCRS([0,-1,0], epoch)
        elif text == "Z" or text == "+Z":
            return _2ND_Orbit.XYZ_to_GCRS([0,0,1], epoch)
        elif text == "-Z":
            return _2ND_Orbit.XYZ_to_GCRS([0,0,-1], epoch)
        elif text == "I":
            return 0
        else:
            try:
                tmp = text.split(',')
                X = float(tmp[0])
                Y = float(tmp[1])
                Z = float(tmp[2])
                tmp = np.array([X,Y,Z])
                tmp = tmp/np.linalg.norm(tmp)
                return _2ND_Orbit.XYZ_to_GCRS(tmp, epoch)
            except Exception as e:
                print("ERROR: During parsing the norm of one node the string is not correct " + str(text) + " '"+str(attitude)+"'")
                exit(1)


def selectFile():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfile(title="Select a file", filetypes=[("All files", "*.*"),("XLSX files", "*.xslx"),("CSV files", "*.csv")])

    return file_path



def main():



    DATA_PROTECTION = True



    SIM_VERSION = "1.0"
    SIM_MAX_PARAM = 29
    SIM_EPS_MAX_PARAM = 12
    FILE_PATH = ""
    FILE_H = ""

    SIM_START_EPOCH = ""


    SIM_MAX_ORBIT = 0.0
    SIM_STEP_SIZE = 0.0

    SIM_MAX_TEMP = 0.0
    SIM_MIN_TEMP = 0.0

    SIM_THERMAL_SIM = False
    SIM_POWER_SIM = False


    SIM_EPS_TOT_CELLS = 0

    SIM_BATTERY_MIN_TEMP = 0.0

    SIM_ATTITUDE = 'R'

    SIM_RAND_COEFF = 0.0

    SIM_R_DOT_CUSTOM = 0.0
    SIM_T_DOT_CUSTOM = 0.0
    SIM_H_DOT_CUSTOM = 0.0


    SIM_PWR_SC_AREA = 0.0
    SIM_PWR_SC_EFF = 0.0
    SIM_PWR_EPS_CHG_EFF = 0.0
    SIM_PWR_EPS_DCHG_EFF = 0.0
    SIM_PWR_SA_EFF = 0.0
    SIM_PWR_BATT_MAX_CAP = 0.0
    SIM_PWR_BATTERY_CAP = 0.0
    SIM_PWR_HEATERS_PWR = 0.0
    SIM_PWR_SC_ALPHA = 0.0
    SIM_PWR_SC_EPS = 0.0
    SIM_PWR_MAX_CHG = 0.0
    SIM_PWR_MAX_DCHG = 0.0


    SIM_ROTATION_VECTOR = []



    SIM_VF_MATRIX = []
    SIM_COND_MATRIX = []
    SIM_COND_DIST_MATRIX = []


    NUM_NODES = 0
    BATTERY_NODE = 0
    Nodes = []





    print("     ------------ 2NDTAN ------------")
    print("Multi-modal thermal and power simulator for cubesats")
    print("Version: " + SIM_VERSION)




    FILE_PATH = selectFile()

    if not FILE_PATH:
        print("ERROR during opening the file")
        exit(1)





    print("Extracting data...")

    print("Reading Nodes parameters sheet...")

    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=0, header=0)
    if FILE_H.empty:
        print("ERROR during reading the NODES sheet")
        exit(1)
    FILE_H = FILE_H.fillna(0)



    NUM_NODES = 0


    # READING FIRST SHEET WITH NODES PROPERTIES
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

        nod.Normal_Nadir = str(FILE_H.loc[i].iat[8])
        nod.Normal_Sun = str(FILE_H.loc[i].iat[7])

        Nodes.append(nod)




    for nod in Nodes:
        SIM_EPS_TOT_CELLS = SIM_EPS_TOT_CELLS + nod.NumCells



    print("\tNumber of nodes: " + str(NUM_NODES))


    # READING SECOND SHEET WITH SIMULATION PARAMETERS
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    Orb = _2ND_Orbit()

    print("Reading Simulation parameters sheet...")

    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=1, header=None)

    if FILE_H.empty:
        print("ERROR: The second sheet of the excel is empty")
        exit(1)

    if len(FILE_H) != SIM_MAX_PARAM:
        print("ERROR: The second sheet with all the simulation parameters is missing some " + str(len(FILE_H)) + " vs " + str(SIM_MAX_PARAM) + " required")
        exit(1)


    index = 0

    try:
        SIM_START_EPOCH = Time(str(FILE_H.iloc[index].iat[1]), scale="utc")
        Orb.Epoch = SIM_START_EPOCH
        index = index + 1
    except Exception as e:
        print("ERROR: The epoch specified is not valid [ROW "+str(index)+"]")
        exit(1)


    Orb.Rearth = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.Rearth < 6000 or Orb.Rearth > 7000) and DATA_PROTECTION:
        print("ERROR: Earth radius is wrong [ROW "+str(index)+"]")
        exit(1)
    
    Orb.Fsun = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.Fsun < 0 or Orb.Fsun > 10000) and DATA_PROTECTION:
        print("ERROR: Sun Flux W/m2 is not realistic [ROW "+str(index)+"]")
        exit(1)

    Orb.albedo = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.albedo < 0 or Orb.albedo > 1) and DATA_PROTECTION:
        print("ERROR: the Earth albedo value is not realistic [ROW "+str(index)+"]")
        eixt(1)

    Orb.Fearth = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.Fearth < 0 or Orb.Fearth > 1000) and DATA_PROTECTION:
        print("ERROR: Earth Flux W/m2 is not realistic [ROW "+str(index)+"]")
        exit(1)


    Orb.alt = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.alt < 200 or Orb.alt > 10000) and DATA_PROTECTION:
        print("ERROR: The orbit altitude in km is too low or high [ROW "+str(index)+"]")
        exit(1)

    Orb.ecc = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.ecc < 0 or Orb.ecc >= 1) and DATA_PROTECTION:
        print("ERROR: the orbit eccentricity is wrong or parabolic/hyperbolic [ROW "+str(index)+"]")
        exit(1)

    Orb.INC = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.INC < 0 or Orb.INC > 180) and DATA_PROTECTION:
        print("ERROR: the orbit inclination is wrong [ROW "+str(index)+"]")
        exit(1)
    Orb.INC = Orb.INC*2*np.pi/360

    Orb.LTAN = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.LTAN < 0 or Orb.LTAN > 24) and DATA_PROTECTION:
        print("ERROR: the LTAN is not valid [ROW "+str(index)+"]")
        exit(1)

    Orb.RAAN = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if Orb.RAAN > 360:
        Orb.RAAN = Orb.RAAN % 360
    if Orb.RAAN < 0 and DATA_PROTECTION:
        print("ERROR: the RAAN is negative [ROW "+str(index)+"]")
        exit(1)
    Orb.RAAN = Orb.RAAN*2*np.pi/360

    Orb.AOP = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if Orb.AOP > 360:
        Orb.AOP = Orb.AOP % 360
    if Orb.AOP < 0 and DATA_PROTECTION:
        print("ERROR: The orbit AOP is negative [ROW "+str(index)+"]")
        exit(1)
    Orb.AOP = Orb.AOP*2*np.pi/360

    Orb.TA = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if Orb.TA > 360:
        Orb.TA = Orb.TA % 360
    if Orb.TA < 0 and DATA_PROTECTION:
        print("ERROR: The orbit TA is negative [ROW "+str(index)+"]")
        exit(1)

    
    SIM_MAX_ORBIT = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if SIM_MAX_ORBIT < 0 and DATA_PROTECTION:
        print("ERROR: Orbits to simulate are negative [ROW "+str(index)+"]")
        exit(1)


    SIM_STEP_SIZE = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if SIM_STEP_SIZE < 0 and DATA_PROTECTION:
        print("ERROR: Simulation step size is negative [ROW "+str(index)+"]")
        exit(1)


    start_temp = float(FILE_H.iloc[index].iat[1]) + 273.15
    index = index + 1
    if (start_temp < 100 or start_temp > 400) and DATA_PROTECTION:
        print("ERROR: the starting temperature of the nodes is too high or low [ROW "+str(index)+"]")
        exit(1)

    for nod in Nodes:
        nod.Temp = start_temp
    
    SIM_MIN_TEMP = float(FILE_H.iloc[index].iat[1]) + 273.15
    index = index + 1
    
    SIM_MAX_TEMP = float(FILE_H.iloc[index].iat[1]) + 273.15
    index = index + 1

    if SIM_MIN_TEMP > SIM_MAX_TEMP and DATA_PROTECTION:
        print("ERROR: The minimum possible temperature is higher than the maximum one [ROW "+str(index-1)+","+str(index)+"]")
        exit(1)

    for nod in Nodes:
        nod.TempMin = SIM_MIN_TEMP
        nod.TempMax = SIM_MAX_TEMP

    try:
        tmp = int(FILE_H.iloc[index].iat[1])
        index = index + 1
        if tmp == 0:
            SIM_THERMAL_SIM = False
        else:
            SIM_THERMAL_SIM = True

        tmp = int(FILE_H.iloc[index].iat[1])
        index = index + 1
        if tmp == 0:
            SIM_POWER_SIM = False
        else:
            SIM_POWER_SIM = True
    except Exception as e:
        print("ERROR: Thermal or Power simulation flag is not valid [ROW "+str(index-1)+","+str(index)+"]")

    
    SIM_BATTERY_MIN_TEMP = float(FILE_H.iloc[index].iat[1]) + 273.15
    index = index + 1
    if (SIM_BATTERY_MIN_TEMP < 200 or SIM_BATTERY_MIN_TEMP > 500) and DATA_PROTECTION:
        print("ERROR: The battery minimum temperature is too large or small [ROW "+str(index)+"]")
        exit(1)

    
    SIM_ATTITUDE = str(FILE_H.iloc[index].iat[1]).upper()
    index = index + 1
    if SIM_ATTITUDE not in ["S","N","R","C","F"]:
        print("ERROR: The starting attitude letter is not valid [ROW "+str(index)+"]")
        exit(1)

    
    SIM_RAND_COEFF = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (SIM_RAND_COEFF < 0 or SIM_RAND_COEFF > 1000) and DATA_PROTECTION:
        print("ERROR: The random coefficient is too large or negative [ROW "+str(index)+"]")
        exit(1)

    
    SIM_R_DOT_CUSTOM = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    SIM_T_DOT_CUSTOM = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    SIM_H_DOT_CUSTOM = float(FILE_H.iloc[index].iat[1])
    index = index + 1

    if abs(SIM_R_DOT_CUSTOM) > 1000 and DATA_PROTECTION:
        print("ERROR: The R dot speed is too large [ROW "+str(index)+"]")
        exit(1)
    if abs(SIM_T_DOT_CUSTOM) > 1000 and DATA_PROTECTION:
        print("ERROR: The T dot speed is too large [ROW "+str(index)+"]")
        exit(1)
    if abs(SIM_H_DOT_CUSTOM) > 1000 and DATA_PROTECTION:
        print("ERROR: The H dot speed is too large [ROW "+str(index)+"]")
        exit(1)

    SIM_DOT_CUSTOM = np.array([SIM_R_DOT_CUSTOM, SIM_T_DOT_CUSTOM, SIM_H_DOT_CUSTOM])

    X_FIXED = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    Y_FIXED = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    Z_FIXED = float(FILE_H.iloc[index].iat[1])
    index = index + 1

    VEC_FIXED = np.array([X_FIXED, Y_FIXED, Z_FIXED])

    if not np.any(VEC_FIXED):
        print("ERROR: The fixed vector is all [0,0,0] [ROW "+str(index-2)+","+str(index-1)+","+str(index)+"]")
        exit(1)

    VEC_FIXED = VEC_FIXED/np.linalg.norm(VEC_FIXED)


    SIM_NODE_POINTING = str(FILE_H.iloc[index].iat[1])
    index = index + 1
    for nod in Nodes:
        if nod.Name == SIM_NODE_POINTING:
            break
    else:
        print("ERROR: The Node name that must point to a fixed point is not present as a node " + str(SIM_NODE_POINTING))
        exit(1)


    
    Orb.create_orbit()


    for nod in Nodes:
        if nod.Normal_Nadir.upper() == "I" and nod.Normal_Sun.upper() == "I":
            nod.Internal = True

        if nod.Internal == True:
            nod.Normal = 0
        else:
            if SIM_ATTITUDE == "N":
                nod.Normal = setup_norm_vector(nod.Normal_Nadir, "N", Orb.Orbit, Orb.Epoch)
            else:
                nod.Normal = setup_norm_vector(nod.Normal_Sun, "S", Orb.Orbit, Orb.Epoch)



    # READING THIRD SHEET WITH EPS SIMULATION PARAMETERS
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    print("Reading EPS parameter sheet...")


    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=2, header=None)

    if FILE_H.empty:
        print("ERROR: The third sheet of the excel is empty")
        exit(1)

    if len(FILE_H) != SIM_EPS_MAX_PARAM:
        print("ERROR: The third sheet with all the EPS parameters is missing some " + str(len(FILE_H)) + " vs " + str(SIM_EPS_MAX_PARAM) + " required")
        exit(1)



    SIM_PWR_SC_AREA = float(FILE_H.iloc[0].iat[1])
    if (SIM_PWR_SC_AREA < 0 or SIM_PWR_SC_AREA > 10000) and DATA_PROTECTION:
        print("ERROR: The solar cell area is negative or too large [ROW 1]")
        exit(1)
    SIM_PWR_SC_AREA = SIM_PWR_SC_AREA/10000


    SIM_PWR_SC_EFF = float(FILE_H.iloc[1].iat[1])
    if (SIM_PWR_SC_EFF < 0 or SIM_PWR_SC_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The solar cell efficiency is negative or > 1 [ROW 2]")
        exit(1)
    

    SIM_PWR_EPS_CHG_EFF = float(FILE_H.iloc[2].iat[1])
    if (SIM_PWR_EPS_CHG_EFF < 0 or SIM_PWR_EPS_CHG_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The charging efficiency is negative or >1 [ROW 3]")
        exit(1)

    SIM_PWR_EPS_DCHG_EFF = float(FILE_H.iloc[3].iat[1])
    if (SIM_PWR_EPS_DCHG_EFF < 0 and SIM_PWR_EPS_DCHG_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The discharge efficiency is negative or >1 [ROW 4]")
        exit(1)

#TODO -> THIS EFFICIENCY WILL HEAT UP THE BATTERY NODE


    SIM_PWR_SA_EFF = float(FILE_H.iloc[4].iat[1])
    if (SIM_PWR_SA_EFF < 0 or SIM_PWR_SA_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The solar array efficiency is negative or >1 [ROW 5]")
        exit(1)
    

#TODO -> THIS EFFICIENCY WILL BE SPREAD TO ALL NODES WITH SOLAR CELLS 


    SIM_PWR_BATT_MAX_CAP = float(FILE_H.iloc[5].iat[1])
    if SIM_PWR_BATT_MAX_CAP < 0 and DATA_PROTECTION:
        print("ERROR: The battery pack max capacity is negative [ROW 6]")
        exit(1)

    
    SIM_PWR_BATTERY_CAP = float(FILE_H.iloc[6].iat[1])
    if (SIM_PWR_BATTERY_CAP < 0 or SIM_PWR_BATTERY_CAP > SIM_PWR_BATT_MAX_CAP) and DATA_PROTECTION:
        print("ERROR: The starting battery capacity is negative or larger than maximum [ROW 7]")
        exit(1)


    SIM_PWR_HEATERS_PWR = float(FILE_H.iloc[7].iat[1])
    if SIM_PWR_HEATERS_PWR < 0 and DATA_PROTECTION:
        print("ERROR: The heaters power is negative [ROW 8]")
        exit(1)


    SIM_PWR_SC_ALPHA = float(FILE_H.iloc[8].iat[1])
    if (SIM_PWR_SC_ALPHA < 0 or SIM_PWR_SC_ALPHA > 1) and DATA_PROTECTION:
        print("ERROR: The alpha coefficient of the cells is negative or >1 [ROW 9]")
        exit(1)
    

    SIM_PWR_SC_EPS = float(FILE_H.iloc[9].iat[1])
    if (SIM_PWR_SC_EPS < 0 or SIM_PWR_SC_EPS > 1) and DATA_PROTECTION:
        print("ERROR: The eps coefficient of the cells is negative or >1 [ROW 10]")
        exit(1)

    
    SIM_PWR_MAX_CHG = float(FILE_H.iloc[10].iat[1])
    if SIM_PWR_MAX_CHG < 0 and DATA_PROTECTION:
        print("ERROR: The maximum charge power is negative [ROW 11]")
        exit(1)
    

    SIM_PWR_MAX_DCHG = float(FILE_H.iloc[11].iat[1])
    if SIM_PWR_MAX_DCHG < 0 and DATA_PROTECTION:
        print("ERROR: The maximum discharge power is negative [ROW 12]")
        exit(1)

    if SIM_PWR_HEATERS_PWR > SIM_PWR_MAX_DCHG and DATA_PROTECTION:
        print("ERROR: The heaters need more power than it is possible to deliver [ROW 8,12]")
        exit(1)



    # READING FOURTH SHEET WITH VISIBLITY FACTOR MATRIX
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




    print("Reading FV sheet...")



    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=3, header=None)
    if FILE_H.empty:
        print("ERROR: The fourth sheet of the excel is empty")
        exit(1)
    FILE_H = FILE_H.fillna(0)
    if FILE_H.shape[0] < NUM_NODES or FILE_H.shape[1] < NUM_NODES:
        print("ERROR: The FV sheet is not large enough")
        exit(1)

    FILE_H = FILE_H.iloc[0:NUM_NODES, 0:NUM_NODES]


    if (FILE_H > 1).any().any():
        print("ERROR: The VF matrix contains at least one element >1")
        exit(1)

    if (FILE_H < 0).any().any():
        print("ERROR: The VF matrix contains at least one element <0")
        exit(1)
    

    SIM_VF_MATRIX = FILE_H.to_numpy(dtype=float)

    SIM_VF_MATRIX = np.triu(SIM_VF_MATRIX) + np.triu(SIM_VF_MATRIX,1).T

    for i in range(0, NUM_NODES):
        if SIM_VF_MATRIX[i,i] != 0:
            print("ERROR: Diagonal component " + str(i+1) + " of the FV matrix is not zero")
            exit(1)


    # READING FIFTH SHEET WITH CONDUCTION AREA BEWTEEN NODES MATRIX
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    print("Reading Conduction area matrix...")

    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=4, header=None)
    if FILE_H.empty:
        print("ERROR: The Conduction area matrix of the excel is empty")
        exit(1)
    FILE_H = FILE_H.fillna(0)
    if FILE_H.shape[0] < NUM_NODES or FILE_H.shape[1] < NUM_NODES:
        print("ERROR: The Conduction area matrix sheet is not large enough")
        exit(1)

    FILE_H = FILE_H.iloc[0:NUM_NODES, 0:NUM_NODES]


    if (FILE_H < 0).any().any():
        print("ERROR: The Conduction area matrix contains at least one element <0")
        exit(1)
    

    SIM_COND_MATRIX = FILE_H.to_numpy(dtype=float)

    SIM_COND_MATRIX = np.triu(SIM_COND_MATRIX) + np.triu(SIM_COND_MATRIX,1).T

    for i in range(0, NUM_NODES):
        if SIM_COND_MATRIX[i,i] != 0:
            print("ERROR: Diagonal component " + str(i+1) + " of the Conduction area matrix is not zero")
            exit(1)

    SIM_COND_MATRIX = SIM_COND_MATRIX/1e6



    # READING SIXTH SHEET WITH DISTANCE BETWEEN NODES FOR CONDUCTION MATRIX
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    print("Reading Conduction distance matrix...")

    FILE_H = pd.read_excel(FILE_PATH.name, sheet_name=5, header=None)
    if FILE_H.empty:
        print("ERROR: The Conduction distance matrix of the excel is empty")
        exit(1)
    FILE_H = FILE_H.fillna(0)
    if FILE_H.shape[0] < NUM_NODES or FILE_H.shape[1] < NUM_NODES:
        print("ERROR: The Conduction distance matrix sheet is not large enough")
        exit(1)

    FILE_H = FILE_H.iloc[0:NUM_NODES, 0:NUM_NODES]

    if (FILE_H < 0).any().any():
        print("ERROR: The Conduction distance matrix contains at least one element <0")
        exit(1)
    

    SIM_COND_DIST_MATRIX = FILE_H.to_numpy(dtype=float)


    for i in range(0, NUM_NODES):
        if SIM_COND_DIST_MATRIX[i,i] != 0:
            print("ERROR: Diagonal component " + str(i+1) + " of the Conduction distance matrix is not zero")
            exit(1)


    SIM_COND_DIST_MATRIX = SIM_COND_DIST_MATRIX/1000


    for ii in range(0,NUM_NODES):
        for jj in range(0, NUM_NODES):
            if (SIM_COND_MATRIX[ii,jj] != 0 and SIM_COND_DIST_MATRIX[ii,jj] == 0) or (SIM_COND_MATRIX[ii,jj] == 0 and SIM_COND_DIST_MATRIX[ii,jj] != 0):
                print("ERROR: Conduction area and Conduction distance matrix have items that are not filled correctly ["+str(ii+1)+","+str(jj+1)+"]")




    # READING SEVENTH SHEET WITH ATTITUDE MISSION TABLE
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




    # READING EIGHTH SHEET WITH INTERNAL POWER NODES MISSION TABLE
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # STARTING THE SIMULATION
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    SIM_ROTATION_VECTOR = set_Rotational_Vector(SIM_ATTITUDE,  SIM_DOT_CUSTOM, SIM_RAND_COEFF, Orb.Orbit)


    print("Starting simulation...")



    SIM_PERCENTAGE = 0
    SIM_PERCENTAGE_STEPUP = 5

    SIM_MAX_STEPS = math.ceil(SIM_MAX_ORBIT*Orb.Orbit.period.to("s").value / SIM_STEP_SIZE)


    TIME_ARRAY = []

    SIM_TIME = 0

    if SIM_ATTITUDE == "N":
        set_Nodes_Nadir(Nodes, SIM_ATTITUDE, Orb.Orbit, SIM_START_EPOCH)
    else:
        set_Nodes_Sun(Nodes, SIM_ATTITUDE, Orb.Orbit, SIM_START_EPOCH)

    SIM_ECLIPSE = 0

    for step in range(0, SIM_MAX_STEPS+1):

        SIM_ECLIPSE = Orb.eclipse()

        if step/SIM_MAX_STEPS*100 > SIM_PERCENTAGE+SIM_PERCENTAGE_STEPUP:
            SIM_PERCENTAGE = SIM_PERCENTAGE + SIM_PERCENTAGE_STEPUP
            print("Simulation working... " + str(SIM_PERCENTAGE) + "% ("+str(step)+"/"+str(SIM_MAX_STEPS)+")")


        # Calculate current time and propagate orbit for accurate positioning
        current_time = SIM_START_EPOCH + step*SIM_STEP_SIZE*u.s
        dt = current_time - Orb.Orbit.epoch
        current_orbit = Orb.Orbit.propagate(dt)


        for nod in Nodes:
            nod.rotate_Normal(SIM_ROTATION_VECTOR, SIM_STEP_SIZE)
            nod.update_angles(current_orbit, current_time)


        #  ORBIT PROPAGATION AND CALCULATIONS OF THE SIMULATION









        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        

        SIM_TIME = SIM_TIME + SIM_STEP_SIZE
        Orb.Epoch = Orb.Epoch + SIM_STEP_SIZE*u.s
        TIME_ARRAY.append(SIM_TIME)


        Orb.propagate_orbit(SIM_STEP_SIZE*u.s)


    # END OF SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    SIM_PERCENTAGE = SIM_PERCENTAGE + SIM_PERCENTAGE_STEPUP
    print("Simulation finished " + str(SIM_PERCENTAGE) + "% ("+str(step)+"/"+str(SIM_MAX_STEPS)+")")


    # DATA MANIPULATION 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   











    # PLOTTING DATA 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for nod in Nodes:
        plt.plot(TIME_ARRAY, nod.SunAngleHistory, label=nod.Name)

    plt.title("Angles")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(TIME_ARRAY, Orb.EclipseHistory)

    plt.show()







if __name__ == "__main__":

    main()