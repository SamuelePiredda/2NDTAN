from _2ND_Orbit import _2ND_Orbit
from _2ND_Nodes import Node


from scipy.optimize import fsolve
import math
import os
import random
import warnings
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


CONST_SIGMA = 5.67e-8



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


def _positive_cos_deg(angle_deg):
    return max(0.0, np.cos(np.deg2rad(angle_deg)))


def _solar_cell_area(node, power_state):
    if power_state is None:
        return 0.0
    return max(0.0, min(node.Area, node.NumCells * power_state["sc_area"]))


def _solar_cell_efficiency(temp_k, power_state):
    if power_state is None:
        return 0.0

    temp_c = temp_k - 273.15
    efficiency = power_state["sc_eff"] + power_state["sc_eff_temp_coeff"] * (temp_c - 25.0)
    return float(np.clip(efficiency, 0.0, 1.0))


def electrical_power_generated_by_node(node, Orbit, power_state):
    if power_state is None or not power_state["enabled"]:
        return 0.0
    if node.Internal or node.NumCells <= 0:
        return 0.0

    cell_area = _solar_cell_area(node, power_state)
    if cell_area <= 0.0:
        return 0.0

    direct_flux = Orbit.Eclipse * Orbit.Fsun * _positive_cos_deg(node.SunAngle)
    albedo_flux = Orbit.Eclipse * Orbit.Fsun * Orbit.albedo * _positive_cos_deg(node.EarthAngle)
    incident_power = cell_area * (direct_flux + albedo_flux)

    if incident_power <= 0.0:
        return 0.0

    return incident_power * _solar_cell_efficiency(node.Temp, power_state)


def evaluate_power_balance(Nodes, Orbit, power_state, dt, battery_cap=None):
    if power_state is None or not power_state["enabled"]:
        return {
            "node_generated": [0.0] * len(Nodes),
            "total_generated_raw": 0.0,
            "total_load": 0.0,
            "total_heater_power": 0.0,
            "solar_bus_power": 0.0,
            "charge_power": 0.0,
            "discharge_power": 0.0,
            "curtailed_power": 0.0,
            "unmet_power": 0.0,
            "battery_loss_power": 0.0,
            "battery_cap_after": battery_cap if battery_cap is not None else 0.0,
        }

    if battery_cap is None:
        battery_cap = power_state["battery_cap"]

    node_generated = []
    total_generated_raw = 0.0
    total_load = 0.0
    total_heater_power = 0.0

    for nod in Nodes:
        generated_power = electrical_power_generated_by_node(nod, Orbit, power_state)
        node_generated.append(generated_power)
        total_generated_raw = total_generated_raw + generated_power
        total_load = total_load + nod.Qint + nod.Qheater
        total_heater_power = total_heater_power + nod.Qheater

    solar_bus_power = total_generated_raw * power_state["sa_eff"]
    battery_node_index = power_state["battery_node"]
    battery_too_cold = (
        battery_node_index is not None
        and Nodes[battery_node_index].Temp < power_state["battery_min_temp"]
    )

    charge_power = 0.0
    discharge_power = 0.0
    curtailed_power = 0.0
    unmet_power = 0.0
    battery_loss_power = 0.0
    battery_cap_after = battery_cap

    net_bus_power = solar_bus_power - total_load

    if net_bus_power >= 0.0:
        requested_charge_power = min(net_bus_power, power_state["max_charge"])

        if not battery_too_cold and power_state["charge_eff"] > 0.0 and dt > 0.0:
            remaining_capacity_wh = max(0.0, power_state["battery_max_cap"] - battery_cap_after)
            charge_limit_from_capacity = remaining_capacity_wh * 3600.0 / (dt * power_state["charge_eff"])
            charge_power = min(requested_charge_power, charge_limit_from_capacity)
            stored_energy_wh = charge_power * power_state["charge_eff"] * dt / 3600.0
            battery_cap_after = min(power_state["battery_max_cap"], battery_cap_after + stored_energy_wh)
            battery_loss_power = charge_power * (1.0 - power_state["charge_eff"])

        curtailed_power = max(0.0, net_bus_power - charge_power)
    else:
        requested_discharge_power = min(-net_bus_power, power_state["max_discharge"])

        if power_state["discharge_eff"] > 0.0 and dt > 0.0:
            discharge_limit_from_capacity = battery_cap_after * power_state["discharge_eff"] * 3600.0 / dt
            discharge_power = min(requested_discharge_power, discharge_limit_from_capacity)
            removed_energy_wh = discharge_power * dt / (3600.0 * power_state["discharge_eff"])
            battery_cap_after = max(0.0, battery_cap_after - removed_energy_wh)
            battery_loss_power = discharge_power * (1.0 / power_state["discharge_eff"] - 1.0)

        unmet_power = max(0.0, -net_bus_power - discharge_power)

    return {
        "node_generated": node_generated,
        "total_generated_raw": total_generated_raw,
        "total_load": total_load,
        "total_heater_power": total_heater_power,
        "solar_bus_power": solar_bus_power,
        "charge_power": charge_power,
        "discharge_power": discharge_power,
        "curtailed_power": curtailed_power,
        "unmet_power": unmet_power,
        "battery_loss_power": battery_loss_power,
        "battery_cap_after": battery_cap_after,
        "battery_too_cold": battery_too_cold,
    }


def set_heater_power_for_step(Nodes, Orbit, power_state):
    for nod in Nodes:
        nod.Qheater = 0.0

    if power_state is None or not power_state["enabled"]:
        return 0.0

    battery_node_index = power_state["battery_node"]
    if battery_node_index is None:
        return 0.0

    if Nodes[battery_node_index].Temp < power_state["battery_min_temp"]:
        base_power_balance = evaluate_power_balance(Nodes, Orbit, power_state, 0.0)
        if base_power_balance["solar_bus_power"] <= base_power_balance["total_load"]:
            return 0.0

        heater_power = power_state["heater_power"]
        Nodes[battery_node_index].Qheater = heater_power
        return heater_power

    return 0.0


def advance_power_simulation(Nodes, Orbit, power_state, dt):
    if power_state is None or not power_state["enabled"]:
        return

    power_balance = evaluate_power_balance(Nodes, Orbit, power_state, dt)

    for i, nod in enumerate(Nodes):
        nod.PowerGenerated = power_balance["node_generated"][i]
        nod.Qsolar_electric = nod.PowerGenerated
        nod.Qbattery_loss = 0.0

    battery_node_index = power_state["battery_node"]
    if battery_node_index is not None:
        Nodes[battery_node_index].Qbattery_loss = power_balance["battery_loss_power"]

    battery_cap_before = power_state["battery_cap"]
    power_state["battery_history"].append(battery_cap_before)
    power_state["battery_cap"] = power_balance["battery_cap_after"]
    power_state["solar_generation_history"].append(power_balance["total_generated_raw"])
    power_state["solar_bus_power_history"].append(power_balance["solar_bus_power"])
    power_state["load_power_history"].append(power_balance["total_load"])
    power_state["heater_power_history"].append(power_balance["total_heater_power"])
    power_state["battery_loss_power_history"].append(power_balance["battery_loss_power"])
    power_state["charge_power_history"].append(power_balance["charge_power"])
    power_state["discharge_power_history"].append(power_balance["discharge_power"])
    power_state["curtailed_power_history"].append(power_balance["curtailed_power"])
    power_state["unmet_power_history"].append(power_balance["unmet_power"])


def dQ(Nodes, Orbit, VF_MATRIX, COND_MATRIX, COND_DIST_MATRIX, dt=0.0, power_state=None):

    for nod in Nodes:
        nod.Qsun = 0.0
        nod.Qalbedo = 0.0
        nod.Qearth = 0.0
        nod.Qspace = 0.0
        nod.Qsolar_electric = 0.0
        nod.Qbattery_loss = 0.0
        nod.PowerGenerated = 0.0
        nod.Q = 0.0

    # SUN AND ALBEDO
    for nod in Nodes:
        if not nod.Internal:
            nod.Qsun = Orbit.Eclipse * Orbit.Fsun * nod.alpha * nod.Area * np.cos(np.deg2rad(nod.SunAngle))
            nod.Qalbedo = Orbit.Eclipse * Orbit.Fsun * Orbit.albedo * nod.alpha * nod.Area * np.cos(np.deg2rad(nod.EarthAngle))
            nod.Q = nod.Q + nod.Qsun + nod.Qalbedo

    power_balance = evaluate_power_balance(Nodes, Orbit, power_state, dt)

    for i, nod in enumerate(Nodes):
        nod.Qsolar_electric = power_balance["node_generated"][i]
        nod.PowerGenerated = nod.Qsolar_electric
        nod.Q = nod.Q - nod.Qsolar_electric

    battery_node_index = None if power_state is None else power_state["battery_node"]
    if battery_node_index is not None:
        Nodes[battery_node_index].Qbattery_loss = power_balance["battery_loss_power"]

    # EARTH INFRARED
    for nod in Nodes:
        if not nod.Internal:
            nod.Qearth = Orbit.Fearth * nod.eps * nod.Area * nod.F
            nod.Q = nod.Q + nod.Qearth

    # EXTERNAL RADIATION TO SPACE/ENVIRONMENT
    for nod in Nodes:
        if not nod.Internal:
            nod.Qspace = -CONST_SIGMA * nod.eps * nod.Area * np.power(nod.Temp, 4)
            nod.Q = nod.Q + nod.Qspace

    # INTERNAL DISSIPATION
    for nod in Nodes:
        nod.Q = nod.Q + nod.Qint + nod.Qheater + nod.Qbattery_loss

    # CONDUCTION
    for row in range(0, len(Nodes)):
        for col in range(0, len(Nodes)):
            if row == col:
                continue
            if COND_MATRIX[row, col] == 0 or COND_DIST_MATRIX[row, col] == 0:
                continue
            if COND_MATRIX[col, row] == 0 or COND_DIST_MATRIX[col, row] == 0:
                continue
            if Nodes[row].k <= 0 or Nodes[col].k <= 0:
                continue

            thermal_res = (
                COND_DIST_MATRIX[row, col] / (Nodes[row].k * COND_MATRIX[row, col])
                + COND_DIST_MATRIX[col, row] / (Nodes[col].k * COND_MATRIX[col, row])
            )

            if thermal_res > 0:
                Nodes[row].Q = Nodes[row].Q + (Nodes[col].Temp - Nodes[row].Temp) / thermal_res

    # IRRADIATION
    for row in range(0, len(Nodes)):
        for col in range(0, len(Nodes)):
            if row == col:
                continue
            if VF_MATRIX[row, col] == 0:
                continue

            Nodes[row].Q = Nodes[row].Q + CONST_SIGMA * Nodes[row].Area * VF_MATRIX[row, col] * Nodes[row].eps * (
                np.power(Nodes[col].Temp, 4) - np.power(Nodes[row].Temp, 4)
            )

    deltaQ = []
    for nod in Nodes:
        deltaQ.append(nod.Q)

    return np.array(deltaQ, dtype=float)


def dTemperature(Nodes, deltaQ, dt):
    deltaT = np.zeros(len(Nodes), dtype=float)

    for i, nod in enumerate(Nodes):
        thermal_capacity = nod.Cs * nod.Mass
        if thermal_capacity <= 0:
            raise ValueError("Thermal capacity must be strictly positive for all nodes")
        deltaT[i] = deltaQ[i] * dt / thermal_capacity

    return deltaT


def _set_node_temperatures(Nodes, temperatures):
    for i, nod in enumerate(Nodes):
        nod.Temp = float(temperatures[i])


def _delta_q_at_temperatures(
    Nodes,
    Orbit,
    VF_MATRIX,
    COND_MATRIX,
    COND_DIST_MATRIX,
    dt,
    temperatures,
    power_state=None,
):
    _set_node_temperatures(Nodes, temperatures)
    return dQ(Nodes, Orbit, VF_MATRIX, COND_MATRIX, COND_DIST_MATRIX, dt, power_state)


def _explicit_substepped_temperature_step(
    Nodes,
    Orbit,
    VF_MATRIX,
    COND_MATRIX,
    COND_DIST_MATRIX,
    dt,
    power_state=None,
    max_substeps=1024,
):
    base_temperatures = np.array([nod.Temp for nod in Nodes], dtype=float)
    substeps = 1

    while substeps <= max_substeps:
        sub_dt = dt / substeps
        temperatures = base_temperatures.copy()
        stable = True

        for _ in range(substeps):
            deltaQ = _delta_q_at_temperatures(
                Nodes,
                Orbit,
                VF_MATRIX,
                COND_MATRIX,
                COND_DIST_MATRIX,
                sub_dt,
                temperatures,
                power_state,
            )
            next_temperatures = temperatures + dTemperature(Nodes, deltaQ, sub_dt)

            if not np.all(np.isfinite(next_temperatures)) or np.any(next_temperatures <= 0.0):
                stable = False
                break

            temperatures = next_temperatures

        if stable:
            return temperatures, substeps

        substeps = substeps * 2

    return base_temperatures, None


def implicit_euler_temperature_step(
    Nodes,
    Orbit,
    VF_MATRIX,
    COND_MATRIX,
    COND_DIST_MATRIX,
    dt,
    power_state=None,
    maxfev=200,
    tol=1e-6,
):
    old_temperatures = np.array([nod.Temp for nod in Nodes], dtype=float)

    def residual(new_temperatures):
        deltaQ = _delta_q_at_temperatures(
            Nodes,
            Orbit,
            VF_MATRIX,
            COND_MATRIX,
            COND_DIST_MATRIX,
            dt,
            new_temperatures,
            power_state,
        )
        return new_temperatures - old_temperatures - dTemperature(Nodes, deltaQ, dt)

    solution, _, ier, message = fsolve(
        residual,
        old_temperatures,
        xtol=tol,
        maxfev=maxfev,
        full_output=True,
    )

    converged = ier == 1 and np.all(np.isfinite(solution)) and np.all(solution > 0.0)

    if not converged:
        _set_node_temperatures(Nodes, old_temperatures)

        solution, substeps_used = _explicit_substepped_temperature_step(
            Nodes,
            Orbit,
            VF_MATRIX,
            COND_MATRIX,
            COND_DIST_MATRIX,
            dt,
            power_state,
        )

        warnings.warn(
            "Implicit Euler thermal solver did not converge"
            + (f" ({message.strip()})" if message else "")
            + "; using an explicit substepped update"
            + (f" with {substeps_used} substeps." if substeps_used is not None else ".")
            + " Consider reducing the simulation step size.",
            RuntimeWarning,
            stacklevel=2,
        )

    _set_node_temperatures(Nodes, old_temperatures)

    for i, nod in enumerate(Nodes):
        nod.updateTemp(solution[i])

    # Refresh the node heat terms with the converged temperatures.
    dQ(Nodes, Orbit, VF_MATRIX, COND_MATRIX, COND_DIST_MATRIX, dt, power_state)


def _shade_eclipse_regions(ax, time_array, eclipse_history):
    eclipse_start = None

    for i, eclipse_value in enumerate(eclipse_history):
        in_eclipse = eclipse_value < 1.0

        if in_eclipse and eclipse_start is None:
            eclipse_start = time_array[i]
        elif not in_eclipse and eclipse_start is not None:
            eclipse_end = time_array[i]
            ax.axvspan(eclipse_start, eclipse_end, color="0.85", alpha=0.5, zorder=0)
            eclipse_start = None

    if eclipse_start is not None and len(time_array) > 0:
        ax.axvspan(eclipse_start, time_array[-1], color="0.85", alpha=0.5, zorder=0)


def plot_temperature_groups(Nodes, time_array, eclipse_history, temp_min_k, temp_max_k):
    groups = sorted({nod.PlotGroup for nod in Nodes})
    temp_min_c = temp_min_k - 273.15
    temp_max_c = temp_max_k - 273.15

    for group in groups:
        fig, ax = plt.subplots()

        for nod in Nodes:
            if nod.PlotGroup == group:
                ax.plot(time_array, np.array(nod.TempHistory) - 273.15, label=nod.Name)

        _shade_eclipse_regions(ax, time_array, eclipse_history)
        ax.axhline(temp_max_c, color="red", linestyle="--", linewidth=1.2, label="TempMax")
        ax.axhline(temp_min_c, color="blue", linestyle="--", linewidth=1.2, label="TempMin")

        ax.set_title(f"Temperature Group {group}")
        ax.set_xlabel("Orbit")
        ax.set_ylabel("Temperature [C]")
        ax.grid(True)
        ax.legend(loc="upper right")


def plot_power_history(time_array, eclipse_history, power_state):
    if power_state is None or not power_state["enabled"]:
        return
    if len(power_state["battery_history"]) != len(time_array):
        return

    fig, (ax_energy, ax_power) = plt.subplots(2, 1, sharex=True)
    battery_history = np.array(power_state["battery_history"], dtype=float)
    battery_max_cap = power_state["battery_max_cap"]

    _shade_eclipse_regions(ax_energy, time_array, eclipse_history)
    ax_energy.plot(time_array, battery_history, label="Battery Energy", color="tab:green")
    ax_energy.axhline(battery_max_cap, color="tab:red", linestyle="--", linewidth=1.2, label="Battery Max")
    ax_energy.set_title("Power Simulation")
    ax_energy.set_ylabel("Energy [Wh]")
    ax_energy.set_ylim(0.0, battery_max_cap + 5.0)
    ax_energy.grid(True)
    ax_energy.legend(loc="upper right")

    _shade_eclipse_regions(ax_power, time_array, eclipse_history)
    ax_power.plot(time_array, power_state["solar_bus_power_history"], label="Solar Bus Power", color="tab:orange")
    ax_power.plot(time_array, power_state["load_power_history"], label="Load Power", color="tab:blue")
    ax_power.plot(time_array, power_state["heater_power_history"], label="Heater Power", color="tab:red")
    ax_power.plot(time_array, power_state["battery_loss_power_history"], label="Battery Loss Heat", color="tab:purple")
    ax_power.set_xlabel("Orbit")
    ax_power.set_ylabel("Power [W]")
    ax_power.grid(True)
    ax_power.legend(loc="upper right")


def plot_battery_soc_history(time_array, eclipse_history, power_state):
    if power_state is None or not power_state["enabled"]:
        return
    if len(power_state["battery_history"]) != len(time_array):
        return

    battery_history = np.array(power_state["battery_history"], dtype=float)
    battery_max_cap = power_state["battery_max_cap"]
    if battery_max_cap > 0.0:
        soc_history = 100.0 * battery_history / battery_max_cap
    else:
        soc_history = np.zeros_like(battery_history)

    fig, ax = plt.subplots()
    _shade_eclipse_regions(ax, time_array, eclipse_history)
    ax.plot(time_array, soc_history, label="State of Charge", color="tab:cyan")
    ax.set_title("Battery State of Charge")
    ax.set_xlabel("Orbit")
    ax.set_ylabel("State of Charge [%]")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True)
    ax.legend(loc="upper right")


def plot_node_generated_power(Nodes, time_array, eclipse_history):
    nodes_with_cells = [nod for nod in Nodes if nod.NumCells > 0]
    if len(nodes_with_cells) == 0:
        return

    fig, ax = plt.subplots()
    _shade_eclipse_regions(ax, time_array, eclipse_history)

    for nod in nodes_with_cells:
        if len(nod.PowerGeneratedHistory) == len(time_array):
            ax.plot(time_array, nod.PowerGeneratedHistory, label=nod.Name)

    ax.set_title("Generated Power by Node")
    ax.set_xlabel("Orbit")
    ax.set_ylabel("Power [W]")
    ax.grid(True)
    ax.legend(loc="upper right")


def _series_to_length(values, target_len):
    values = list(values)
    if len(values) >= target_len:
        return values[:target_len]
    return values + [np.nan] * (target_len - len(values))


def _safe_filename(text):
    safe_chars = []
    for char in str(text):
        if char.isalnum() or char in ["-", "_"]:
            safe_chars.append(char)
        elif char in [" ", ".", "(", ")"]:
            safe_chars.append("_")
    safe_text = "".join(safe_chars).strip("_")
    return safe_text if safe_text else "figure"


def export_simulation_csv(output_dir, base_name, orbit_array, time_array, eclipse_history, attitude_history, Nodes, power_state):
    target_len = len(orbit_array)
    data = {
        "orbit": _series_to_length(orbit_array, target_len),
        "time_s": _series_to_length(time_array, target_len),
        "eclipse": _series_to_length(eclipse_history, target_len),
        "attitude": _series_to_length(attitude_history, target_len),
    }

    for nod in Nodes:
        node_name = _safe_filename(nod.Name)
        temp_k = _series_to_length(nod.TempHistory, target_len)
        data[f"{node_name}_temp_K"] = temp_k
        data[f"{node_name}_temp_C"] = [value - 273.15 if pd.notna(value) else np.nan for value in temp_k]
        data[f"{node_name}_sun_angle_deg"] = _series_to_length(nod.SunAngleHistory, target_len)
        data[f"{node_name}_earth_angle_deg"] = _series_to_length(nod.EarthAngleHistory, target_len)
        data[f"{node_name}_internal_power_W"] = _series_to_length(nod.InternalPowerHistory, target_len)
        data[f"{node_name}_generated_power_W"] = _series_to_length(nod.PowerGeneratedHistory, target_len)
        data[f"{node_name}_heater_power_W"] = _series_to_length(nod.HeaterPowerHistory, target_len)
        data[f"{node_name}_battery_loss_power_W"] = _series_to_length(nod.BatteryLossHistory, target_len)

    if power_state is not None and power_state["enabled"]:
        battery_history = _series_to_length(power_state["battery_history"], target_len)
        battery_max_cap = power_state["battery_max_cap"]
        data["battery_energy_Wh"] = battery_history
        if battery_max_cap > 0.0:
            data["battery_soc_pct"] = [100.0 * value / battery_max_cap if pd.notna(value) else np.nan for value in battery_history]
        else:
            data["battery_soc_pct"] = [0.0] * target_len
        data["solar_generation_raw_W"] = _series_to_length(power_state["solar_generation_history"], target_len)
        data["solar_bus_power_W"] = _series_to_length(power_state["solar_bus_power_history"], target_len)
        data["load_power_W"] = _series_to_length(power_state["load_power_history"], target_len)
        data["heater_power_total_W"] = _series_to_length(power_state["heater_power_history"], target_len)
        data["battery_loss_power_total_W"] = _series_to_length(power_state["battery_loss_power_history"], target_len)
        data["charge_power_W"] = _series_to_length(power_state["charge_power_history"], target_len)
        data["discharge_power_W"] = _series_to_length(power_state["discharge_power_history"], target_len)
        data["curtailed_power_W"] = _series_to_length(power_state["curtailed_power_history"], target_len)
        data["unmet_power_W"] = _series_to_length(power_state["unmet_power_history"], target_len)

    pd.DataFrame(data).to_csv(os.path.join(output_dir, base_name + "_data.csv"), index=False)


def save_all_figures(output_dir):
    used_names = set()

    for fig_number in plt.get_fignums():
        fig = plt.figure(fig_number)
        title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        if title == "" and len(fig.axes) > 0:
            title = fig.axes[0].get_title()
        file_name = _safe_filename(title if title else f"figure_{fig_number}")

        if file_name in used_names:
            suffix = 2
            while f"{file_name}_{suffix}" in used_names:
                suffix = suffix + 1
            file_name = f"{file_name}_{suffix}"

        used_names.add(file_name)
        fig.savefig(os.path.join(output_dir, file_name + ".png"), dpi=300, bbox_inches="tight")


def parse_fixed_pointing_vector(text):
    tmp = str(text).split(',')
    if len(tmp) != 3:
        raise ValueError("Fixed pointing vector must be in the form X,Y,Z")

    vector = np.array([float(tmp[0]), float(tmp[1]), float(tmp[2])], dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Fixed pointing vector cannot be [0,0,0]")

    return vector / norm


def read_attitude_mission_sheet(file_path, data_protection):
    mission_state = {
        "events": [],
        "repeat": False,
    }

    try:
        file_h = pd.read_excel(file_path, sheet_name=6, header=None).fillna("")
    except Exception:
        return mission_state

    if file_h.empty:
        return mission_state

    for i in range(len(file_h)):
        raw_time = str(file_h.iloc[i].iat[0]).strip()
        if raw_time == "":
            continue

        raw_time_upper = raw_time.upper()
        if raw_time_upper == "END":
            break
        if raw_time_upper == "REP":
            mission_state["repeat"] = True
            break

        try:
            orbit_unit = float(raw_time)
        except ValueError:
            continue

        raw_attitude = str(file_h.iloc[i].iat[1]).strip().upper()
        if raw_attitude not in ["S", "N", "C", "R", "F"]:
            if data_protection:
                raise ValueError("Invalid attitude in mission sheet at row " + str(i + 1))
            continue

        fixed_vector = None
        if raw_attitude == "F":
            fixed_vector = parse_fixed_pointing_vector(file_h.iloc[i].iat[2])

        mission_state["events"].append(
            {
                "orbit_unit": orbit_unit,
                "attitude": raw_attitude,
                "fixed_vector": fixed_vector,
            }
        )

    mission_state["events"].sort(key=lambda event: event["orbit_unit"])
    return mission_state


def read_power_mission_sheet(file_path, node_names, data_protection):
    mission_state = {
        "events": [],
        "repeat": False,
    }

    try:
        file_h = pd.read_excel(file_path, sheet_name=7, header=None).fillna("")
    except Exception:
        return mission_state

    if file_h.empty:
        return mission_state

    for i in range(len(file_h)):
        raw_time = str(file_h.iloc[i].iat[0]).strip()
        if raw_time == "":
            continue

        raw_time_upper = raw_time.upper()
        if raw_time_upper == "END":
            break
        if raw_time_upper == "REP":
            mission_state["repeat"] = True
            break

        try:
            orbit_unit = float(raw_time)
        except ValueError:
            continue

        node_name = str(file_h.iloc[i].iat[1]).strip()
        if node_name not in node_names:
            if data_protection:
                raise ValueError("Invalid node name in power mission sheet at row " + str(i + 1))
            continue

        node_power = float(file_h.iloc[i].iat[2])
        if node_power < 0.0 and data_protection:
            raise ValueError("Negative internal power in power mission sheet at row " + str(i + 1))

        mission_state["events"].append(
            {
                "orbit_unit": orbit_unit,
                "node_name": node_name,
                "power": node_power,
            }
        )

    mission_state["events"].sort(key=lambda event: event["orbit_unit"])
    return mission_state


def get_mission_attitude_state(orbit_unit, default_attitude, default_fixed_vector, mission_state):
    if mission_state is None or len(mission_state["events"]) == 0:
        return default_attitude, default_fixed_vector

    active_event = None

    if mission_state["repeat"]:
        phase = orbit_unit % 1.0
        for event in mission_state["events"]:
            if event["orbit_unit"] <= phase:
                active_event = event
            else:
                break

        if active_event is None and orbit_unit >= 1.0:
            active_event = mission_state["events"][-1]
    else:
        for event in mission_state["events"]:
            if event["orbit_unit"] <= orbit_unit:
                active_event = event
            else:
                break

    if active_event is None:
        return default_attitude, default_fixed_vector

    if active_event["attitude"] == "F" and active_event["fixed_vector"] is not None:
        return active_event["attitude"], active_event["fixed_vector"]

    return active_event["attitude"], default_fixed_vector


def get_mission_power_state(orbit_unit, default_node_power, mission_state):
    if mission_state is None or len(mission_state["events"]) == 0:
        return dict(default_node_power)

    power_state = dict(default_node_power)

    if mission_state["repeat"] and orbit_unit >= 1.0:
        for event in mission_state["events"]:
            power_state[event["node_name"]] = event["power"]
        phase = orbit_unit % 1.0
    else:
        phase = orbit_unit

    for event in mission_state["events"]:
        if event["orbit_unit"] <= phase:
            power_state[event["node_name"]] = event["power"]
        else:
            break

    return power_state


def apply_power_mission_to_nodes(Nodes, node_power_state):
    for nod in Nodes:
        if nod.Name in node_power_state:
            nod.Qint = node_power_state[nod.Name]


def fixed_xyz_to_gcrs(vector, fixed_direction):
    x_dot = np.array(fixed_direction, dtype=float)
    x_dot = x_dot / np.linalg.norm(x_dot)

    z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(x_dot, z_ref)) > 0.99:
        z_ref = np.array([0.0, 1.0, 0.0], dtype=float)

    y_dot = np.cross(z_ref, x_dot)
    y_dot = y_dot / np.linalg.norm(y_dot)

    z_dot = np.cross(x_dot, y_dot)
    z_dot = z_dot / np.linalg.norm(z_dot)

    m_xyz_to_gcrs = np.vstack((x_dot, y_dot, z_dot)).T
    return m_xyz_to_gcrs @ np.array(vector, dtype=float)


def print_attitude_rotation_info(attitude, rotation_vector, custom_vector):
    if attitude == "R":
        print(
            "Random attitude rotation [deg/s]: "
            + f"X={rotation_vector[0]:.3f}, "
            + f"Y={rotation_vector[1]:.3f}, "
            + f"Z={rotation_vector[2]:.3f}"
        )
    elif attitude == "C":
        print(
            "Custom attitude rotation command [deg/s]: "
            + f"R={custom_vector[0]:.3f}, "
            + f"T={custom_vector[1]:.3f}, "
            + f"H={custom_vector[2]:.3f}"
        )



def set_Rotational_Vector(attitude, custom, random_coeff, orb):

    if attitude == "S":
        return np.array([0,0,0])
    elif attitude == "N":
        return _2ND_Orbit.get_nadir_rotation_vector(orb)
    elif attitude == "R":
        X = (random.random() - 0.5)*2*random_coeff
        Y = (random.random() - 0.5)*2*random_coeff
        Z = (random.random() - 0.5)*2*random_coeff
        return np.array([X,Y,Z])
    elif attitude == "C":
        return _2ND_Orbit.RTH_to_GCRS(custom, orb)
    # INTERTIAL POINTING
    elif attitude == "F":
        return np.array([0,0,0])
    else:
        print("ERROR: error during setting the rotataional vector with attitude " + str(attitude))
        exit(1)




def update_Node_Attitude(node, attitude, orb, epoch, fixed_vector=None):
    if node.Internal:
        return

    if attitude == "N":
        node.Normal = setup_norm_vector(node.Normal_Nadir, "N", orb, epoch)
    elif attitude == "S":
        node.Normal = setup_norm_vector(node.Normal_Sun, "S", orb, epoch)
    elif attitude == "F":
        node.Normal = setup_norm_vector(node.Normal_Sun, "F", orb, epoch, fixed_vector)


def set_Nodes_Nadir(Nodes, attitude, orb, epoch):
    for nod in Nodes:
        nod.Normal = setup_norm_vector(nod.Normal_Nadir, attitude, orb, epoch)

def set_Nodes_Sun(Nodes, attitude, orb, epoch, fixed_vector=None):
    for nod in Nodes:
        nod.Normal = setup_norm_vector(nod.Normal_Sun, attitude, orb, epoch, fixed_vector)


def setup_norm_vector(text, attitude, orb, epoch, fixed_vector=None):

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

    elif attitude == "S":
        if text == "X" or text == "+X":
            return _2ND_Orbit.SUNXYZ_to_GCRS([1,0,0], epoch)
        elif text == "-X":
            return _2ND_Orbit.SUNXYZ_to_GCRS([-1,0,0], epoch)
        elif text == "Y" or text == "+Y":
            return _2ND_Orbit.SUNXYZ_to_GCRS([0,1,0], epoch)
        elif text == "-Y":
            return _2ND_Orbit.SUNXYZ_to_GCRS([0,-1,0], epoch)
        elif text == "Z" or text == "+Z":
            return _2ND_Orbit.SUNXYZ_to_GCRS([0,0,1], epoch)
        elif text == "-Z":
            return _2ND_Orbit.SUNXYZ_to_GCRS([0,0,-1], epoch)
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
                return _2ND_Orbit.SUNXYZ_to_GCRS(tmp, epoch)
            except Exception as e:
                print("ERROR: During parsing the norm of one node the string is not correct " + str(text) + " '"+str(attitude)+"'")
                exit(1)

    elif attitude == "F":
        if text == "X" or text == "+X":
            return fixed_xyz_to_gcrs([1,0,0], fixed_vector)
        elif text == "-X":
            return fixed_xyz_to_gcrs([-1,0,0], fixed_vector)
        elif text == "Y" or text == "+Y":
            return fixed_xyz_to_gcrs([0,1,0], fixed_vector)
        elif text == "-Y":
            return fixed_xyz_to_gcrs([0,-1,0], fixed_vector)
        elif text == "Z" or text == "+Z":
            return fixed_xyz_to_gcrs([0,0,1], fixed_vector)
        elif text == "-Z":
            return fixed_xyz_to_gcrs([0,0,-1], fixed_vector)
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
                return fixed_xyz_to_gcrs(tmp, fixed_vector)
            except Exception as e:
                print("ERROR: During parsing the norm of one node the string is not correct " + str(text) + " '"+str(attitude)+"'")
                exit(1)
    else:
        if text == "X" or text == "+X":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([1,0,0])
        elif text == "-X":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([-1,0,0])
        elif text == "Y" or text == "+Y":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([0,1,0])
        elif text == "-Y":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([0,-1,0])
        elif text == "Z" or text == "+Z":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([0,0,1])
        elif text == "-Z":
            return _2ND_Orbit.INERTIALXYZ_to_GCRS([0,0,-1])
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
                return _2ND_Orbit.INERTIALXYZ_to_GCRS(tmp)
            except Exception as e:
                print("ERROR: During parsing the norm of one node the string is not correct " + str(text) + " '"+str(attitude)+"'")
                exit(1)


def selectFile():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfile(title="Select a file", filetypes=[("All files", "*.*"),("XLSX files", "*.xlsx"),("CSV files", "*.csv")])

    return file_path



def main():



    DATA_PROTECTION = True



    SIM_VERSION = "1.0"
    SIM_MAX_PARAM = 29
    SIM_EPS_MAX_PARAM = 13
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
    SIM_PWR_SC_EFF_TEMP_COEFF = 0.0
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
    ATTITUDE_MISSION = {
        "events": [],
        "repeat": False,
    }
    POWER_MISSION = {
        "events": [],
        "repeat": False,
    }
    DEFAULT_NODE_POWER = {}
    POWER_STATE = None


    NUM_NODES = 0
    BATTERY_NODE = None
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


        nod.k = float(FILE_H.loc[i].iat[5])
        if nod.k < 0.0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative thermal conductivity")
            exit(1)

        nod.Area = float(FILE_H.loc[i].iat[6])/10000
        if nod.Area < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative exposed area")
            exit(1)

        nod.Qint = float(FILE_H.loc[i].iat[7])

        nod.PlotGroup = int(FILE_H.loc[i].iat[10])

        nod.NumCells = int(FILE_H.loc[i].iat[11])
        if nod.NumCells < 0 and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has negative number of associated cells")
            exit(1)

        battery_flag = int(FILE_H.loc[i].iat[12])
        if battery_flag not in [0, 1] and DATA_PROTECTION:
            print("ERROR: the node " + nod.Name + " has an invalid battery flag")
            exit(1)
        if battery_flag == 1:
            if BATTERY_NODE is not None and DATA_PROTECTION:
                print("ERROR: more than one battery node is defined")
                exit(1)
            BATTERY_NODE = len(Nodes)
            nod.BatteryNode = True

        nod.Normal_Nadir = str(FILE_H.loc[i].iat[9])
        nod.Normal_Sun = str(FILE_H.loc[i].iat[8])

        Nodes.append(nod)




    for nod in Nodes:
        SIM_EPS_TOT_CELLS = SIM_EPS_TOT_CELLS + nod.NumCells

    DEFAULT_NODE_POWER = {nod.Name: nod.Qint for nod in Nodes}



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
        exit(1)

    Orb.Fearth = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.Fearth < 0 or Orb.Fearth > 1000) and DATA_PROTECTION:
        print("ERROR: Earth Flux W/m2 is not realistic [ROW "+str(index)+"]")
        exit(1)


    Orb.alt = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if (Orb.alt < 200 or Orb.alt > 10000) and DATA_PROTECTION:
        print("ERROR: The pericenter altitude in km is too low or high [ROW "+str(index)+"]")
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

    raw_ltan = FILE_H.iloc[index].iat[1]
    index = index + 1
    if pd.isna(raw_ltan):
        Orb.LTAN = None
    else:
        Orb.LTAN = float(raw_ltan)
    if Orb.LTAN is not None and (Orb.LTAN < 0 or Orb.LTAN > 24) and DATA_PROTECTION:
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
    Orb.TA = Orb.TA*2*np.pi/360

    
    SIM_MAX_ORBIT = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if SIM_MAX_ORBIT < 0 and DATA_PROTECTION:
        print("ERROR: Orbits to simulate are negative [ROW "+str(index)+"]")
        exit(1)


    SIM_STEP_SIZE = float(FILE_H.iloc[index].iat[1])
    index = index + 1
    if SIM_STEP_SIZE <= 0 and DATA_PROTECTION:
        print("ERROR: Simulation step size must be strictly positive [ROW "+str(index)+"]")
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

    SIM_PWR_SC_EFF_TEMP_COEFF = float(FILE_H.iloc[2].iat[1])

    SIM_PWR_EPS_CHG_EFF = float(FILE_H.iloc[3].iat[1])
    if (SIM_PWR_EPS_CHG_EFF < 0 or SIM_PWR_EPS_CHG_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The charging efficiency is negative or >1 [ROW 4]")
        exit(1)

    SIM_PWR_EPS_DCHG_EFF = float(FILE_H.iloc[4].iat[1])
    if (SIM_PWR_EPS_DCHG_EFF < 0 or SIM_PWR_EPS_DCHG_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The discharge efficiency is negative or >1 [ROW 5]")
        exit(1)

#TODO -> THIS EFFICIENCY WILL HEAT UP THE BATTERY NODE


    SIM_PWR_SA_EFF = float(FILE_H.iloc[5].iat[1])
    if (SIM_PWR_SA_EFF < 0 or SIM_PWR_SA_EFF > 1) and DATA_PROTECTION:
        print("ERROR: The solar array efficiency is negative or >1 [ROW 6]")
        exit(1)
    

#TODO -> THIS EFFICIENCY WILL BE SPREAD TO ALL NODES WITH SOLAR CELLS 


    SIM_PWR_BATT_MAX_CAP = float(FILE_H.iloc[6].iat[1])
    if SIM_PWR_BATT_MAX_CAP < 0 and DATA_PROTECTION:
        print("ERROR: The battery pack max capacity is negative [ROW 7]")
        exit(1)

    
    SIM_PWR_BATTERY_CAP = float(FILE_H.iloc[7].iat[1])
    if (SIM_PWR_BATTERY_CAP < 0 or SIM_PWR_BATTERY_CAP > SIM_PWR_BATT_MAX_CAP) and DATA_PROTECTION:
        print("ERROR: The starting battery capacity is negative or larger than maximum [ROW 8]")
        exit(1)


    SIM_PWR_HEATERS_PWR = float(FILE_H.iloc[8].iat[1])
    if SIM_PWR_HEATERS_PWR < 0 and DATA_PROTECTION:
        print("ERROR: The heaters power is negative [ROW 9]")
        exit(1)


    SIM_PWR_SC_ALPHA = float(FILE_H.iloc[9].iat[1])
    if (SIM_PWR_SC_ALPHA < 0 or SIM_PWR_SC_ALPHA > 1) and DATA_PROTECTION:
        print("ERROR: The alpha coefficient of the cells is negative or >1 [ROW 10]")
        exit(1)
    

    SIM_PWR_SC_EPS = float(FILE_H.iloc[10].iat[1])
    if (SIM_PWR_SC_EPS < 0 or SIM_PWR_SC_EPS > 1) and DATA_PROTECTION:
        print("ERROR: The eps coefficient of the cells is negative or >1 [ROW 11]")
        exit(1)

    for nod in Nodes:
        if nod.NumCells <= 0 or nod.Area <= 0.0:
            continue

        solar_cell_area = max(0.0, min(nod.Area, nod.NumCells * SIM_PWR_SC_AREA))
        if solar_cell_area <= 0.0:
            continue

        base_area = nod.Area - solar_cell_area
        nod.alpha = (nod.alpha * base_area + SIM_PWR_SC_ALPHA * solar_cell_area) / nod.Area
        nod.eps = (nod.eps * base_area + SIM_PWR_SC_EPS * solar_cell_area) / nod.Area

    
    SIM_PWR_MAX_CHG = float(FILE_H.iloc[11].iat[1])
    if SIM_PWR_MAX_CHG < 0 and DATA_PROTECTION:
        print("ERROR: The maximum charge power is negative [ROW 12]")
        exit(1)
    

    SIM_PWR_MAX_DCHG = float(FILE_H.iloc[12].iat[1])
    if SIM_PWR_MAX_DCHG < 0 and DATA_PROTECTION:
        print("ERROR: The maximum discharge power is negative [ROW 13]")
        exit(1)

    if SIM_PWR_HEATERS_PWR > SIM_PWR_MAX_DCHG and DATA_PROTECTION:
        print("ERROR: The heaters need more power than it is possible to deliver [ROW 9,13]")
        exit(1)

    POWER_STATE = {
        "enabled": SIM_POWER_SIM,
        "sc_area": SIM_PWR_SC_AREA,
        "sc_eff": SIM_PWR_SC_EFF,
        "sc_eff_temp_coeff": SIM_PWR_SC_EFF_TEMP_COEFF,
        "sa_eff": SIM_PWR_SA_EFF,
        "charge_eff": SIM_PWR_EPS_CHG_EFF,
        "discharge_eff": SIM_PWR_EPS_DCHG_EFF,
        "battery_max_cap": SIM_PWR_BATT_MAX_CAP,
        "battery_cap": SIM_PWR_BATTERY_CAP,
        "heater_power": SIM_PWR_HEATERS_PWR,
        "battery_min_temp": SIM_BATTERY_MIN_TEMP,
        "max_charge": SIM_PWR_MAX_CHG,
        "max_discharge": SIM_PWR_MAX_DCHG,
        "battery_node": BATTERY_NODE,
        "battery_history": [],
        "solar_generation_history": [],
        "solar_bus_power_history": [],
        "load_power_history": [],
        "heater_power_history": [],
        "battery_loss_power_history": [],
        "charge_power_history": [],
        "discharge_power_history": [],
        "curtailed_power_history": [],
        "unmet_power_history": [],
    }

    if SIM_POWER_SIM and BATTERY_NODE is None:
        warnings.warn(
            "Power simulation is enabled but no battery node is flagged in the nodes sheet. "
            "Battery energy will still be tracked, but heater thermal coupling is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )



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
                exit(1)




    # READING SEVENTH SHEET WITH ATTITUDE MISSION TABLE
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("Reading Attitude mission sheet...")

    try:
        ATTITUDE_MISSION = read_attitude_mission_sheet(FILE_PATH.name, DATA_PROTECTION)
    except Exception as e:
        print("ERROR: The attitude mission sheet is not valid")
        exit(1)

    print(
        "\tAttitude mission events: "
        + str(len(ATTITUDE_MISSION["events"]))
        + (" (repeat every orbit)" if ATTITUDE_MISSION["repeat"] else "")
    )




    # READING EIGHTH SHEET WITH INTERNAL POWER NODES MISSION TABLE
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("Reading Power mission sheet...")

    try:
        POWER_MISSION = read_power_mission_sheet(
            FILE_PATH.name,
            {nod.Name for nod in Nodes},
            DATA_PROTECTION,
        )
    except Exception as e:
        print("ERROR: The power mission sheet is not valid")
        exit(1)

    print(
        "\tPower mission events: "
        + str(len(POWER_MISSION["events"]))
        + (" (repeat every orbit)" if POWER_MISSION["repeat"] else "")
    )






    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # STARTING THE SIMULATION
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    SIM_ROTATION_VECTOR = np.array([0.0,0.0,0.0])


    print("Starting simulation...")
    if BATTERY_NODE is None:
        print("Battery node present: NO")
    else:
        print("Battery node present: YES (" + Nodes[BATTERY_NODE].Name + ")")



    SIM_PERCENTAGE = 0
    SIM_PERCENTAGE_STEPUP = 5

    ORBIT_PERIOD_S = Orb.Orbit.period.to("s").value
    SIM_MAX_STEPS = math.ceil(SIM_MAX_ORBIT*ORBIT_PERIOD_S / SIM_STEP_SIZE)

    TIME_ARRAY = []
    ORBIT_ARRAY = []
    ATTITUDE_HISTORY = []

    initial_orbit = Orb.Orbit
    CURRENT_ATTITUDE, CURRENT_FIXED_VECTOR = get_mission_attitude_state(
        0.0,
        SIM_ATTITUDE,
        VEC_FIXED,
        ATTITUDE_MISSION,
    )
    PREVIOUS_ATTITUDE = CURRENT_ATTITUDE

    if CURRENT_ATTITUDE == "N":
        set_Nodes_Nadir(Nodes, CURRENT_ATTITUDE, Orb.Orbit, SIM_START_EPOCH)
    elif CURRENT_ATTITUDE == "F":
        set_Nodes_Sun(Nodes, CURRENT_ATTITUDE, Orb.Orbit, SIM_START_EPOCH, CURRENT_FIXED_VECTOR)
    else:
        set_Nodes_Sun(Nodes, "S", Orb.Orbit, SIM_START_EPOCH)

    if CURRENT_ATTITUDE in ["R", "C"]:
        SIM_ROTATION_VECTOR = set_Rotational_Vector(CURRENT_ATTITUDE, SIM_DOT_CUSTOM, SIM_RAND_COEFF, Orb.Orbit)
        print_attitude_rotation_info(CURRENT_ATTITUDE, SIM_ROTATION_VECTOR, SIM_DOT_CUSTOM)

    SIM_ECLIPSE = 0

    for step in range(0, SIM_MAX_STEPS+1):

        current_time = SIM_START_EPOCH + step*SIM_STEP_SIZE*u.s
        current_orbit = initial_orbit.propagate(step*SIM_STEP_SIZE*u.s)

        Orb.Epoch = current_time
        Orb.Orbit = current_orbit

        SIM_TIME = step*SIM_STEP_SIZE
        TIME_ARRAY.append(SIM_TIME)
        CURRENT_ORBIT_UNIT = SIM_TIME / ORBIT_PERIOD_S
        ORBIT_ARRAY.append(CURRENT_ORBIT_UNIT)
        CURRENT_ATTITUDE, CURRENT_FIXED_VECTOR = get_mission_attitude_state(
            CURRENT_ORBIT_UNIT,
            SIM_ATTITUDE,
            VEC_FIXED,
            ATTITUDE_MISSION,
        )
        ATTITUDE_HISTORY.append(CURRENT_ATTITUDE)
        CURRENT_NODE_POWER = get_mission_power_state(
            CURRENT_ORBIT_UNIT,
            DEFAULT_NODE_POWER,
            POWER_MISSION,
        )
        apply_power_mission_to_nodes(Nodes, CURRENT_NODE_POWER)

        if CURRENT_ATTITUDE != PREVIOUS_ATTITUDE:
            if CURRENT_ATTITUDE == "R":
                SIM_ROTATION_VECTOR = set_Rotational_Vector(CURRENT_ATTITUDE, SIM_DOT_CUSTOM, SIM_RAND_COEFF, current_orbit)
                print_attitude_rotation_info(CURRENT_ATTITUDE, SIM_ROTATION_VECTOR, SIM_DOT_CUSTOM)
            elif CURRENT_ATTITUDE == "C":
                SIM_ROTATION_VECTOR = set_Rotational_Vector(CURRENT_ATTITUDE, SIM_DOT_CUSTOM, SIM_RAND_COEFF, current_orbit)
                print_attitude_rotation_info(CURRENT_ATTITUDE, SIM_ROTATION_VECTOR, SIM_DOT_CUSTOM)
            else:
                SIM_ROTATION_VECTOR = np.array([0.0,0.0,0.0])
            PREVIOUS_ATTITUDE = CURRENT_ATTITUDE

        SIM_ECLIPSE = Orb.eclipse()

        if SIM_MAX_STEPS != 0 and step/SIM_MAX_STEPS*100 > SIM_PERCENTAGE+SIM_PERCENTAGE_STEPUP:
            SIM_PERCENTAGE = SIM_PERCENTAGE + SIM_PERCENTAGE_STEPUP
            print("Simulation working... " + str(SIM_PERCENTAGE) + "% ("+str(step)+"/"+str(SIM_MAX_STEPS)+")")


        for nod in Nodes:
            
            update_Node_Attitude(nod, CURRENT_ATTITUDE, current_orbit, current_time, CURRENT_FIXED_VECTOR)

            if CURRENT_ATTITUDE in ["C", "R"] and step > 0:
                if CURRENT_ATTITUDE == "C":
                    SIM_ROTATION_VECTOR = set_Rotational_Vector(CURRENT_ATTITUDE, SIM_DOT_CUSTOM, SIM_RAND_COEFF, current_orbit)
                nod.rotate_Normal(SIM_ROTATION_VECTOR, SIM_STEP_SIZE)

            nod.update_angles(Orb, current_time)


        #  ORBIT PROPAGATION AND CALCULATIONS OF THE SIMULATION

        set_heater_power_for_step(Nodes, Orb, POWER_STATE)

        if SIM_THERMAL_SIM:
            implicit_euler_temperature_step(
                Nodes,
                Orb,
                SIM_VF_MATRIX,
                SIM_COND_MATRIX,
                SIM_COND_DIST_MATRIX,
                SIM_STEP_SIZE,
                POWER_STATE,
            )

        if SIM_POWER_SIM:
            advance_power_simulation(
                Nodes,
                Orb,
                POWER_STATE,
                SIM_STEP_SIZE,
            )





        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        

    # END OF SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    SIM_PERCENTAGE = SIM_PERCENTAGE + SIM_PERCENTAGE_STEPUP
    print("Simulation finished " + str(SIM_PERCENTAGE) + "% ("+str(step)+"/"+str(SIM_MAX_STEPS)+")")


    # DATA MANIPULATION 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    excel_dir = os.path.dirname(FILE_PATH.name)
    excel_base_name = os.path.splitext(os.path.basename(FILE_PATH.name))[0]
    output_dir = os.path.join(excel_dir, excel_base_name + "output")
    os.makedirs(output_dir, exist_ok=True)

    export_simulation_csv(
        output_dir,
        excel_base_name,
        ORBIT_ARRAY,
        TIME_ARRAY,
        Orb.EclipseHistory,
        ATTITUDE_HISTORY,
        Nodes,
        POWER_STATE,
    )
    print("CSV data saved in " + output_dir)











    # PLOTTING DATA 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if SIM_THERMAL_SIM:
        plot_temperature_groups(
            Nodes,
            ORBIT_ARRAY,
            Orb.EclipseHistory,
            SIM_MIN_TEMP,
            SIM_MAX_TEMP,
        )

    if SIM_POWER_SIM:
        plot_power_history(
            ORBIT_ARRAY,
            Orb.EclipseHistory,
            POWER_STATE,
        )
        plot_battery_soc_history(
            ORBIT_ARRAY,
            Orb.EclipseHistory,
            POWER_STATE,
        )
        plot_node_generated_power(
            Nodes,
            ORBIT_ARRAY,
            Orb.EclipseHistory,
        )

    save_all_figures(output_dir)
    print("PNG figures saved in " + output_dir)

    plt.show()







if __name__ == "__main__":

    main()
