"""Simulation utilities for nuclear reaction and gamma-ray spectroscopy analysis.

This module provides tools for simulating nuclear reaction yields, calculating
gamma-ray counts, and generating projected spectra based on nuclear level schemes.
"""

# pylint: disable=invalid-name
# pylint: disable=consider-using-from-import

import ast
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import ROOT  # pylint: disable=import-error
from scipy.stats import norm  # pylint: disable=import-error
from tqdm.notebook import tqdm  # pylint: disable=import-error

from gammaspy.nudat import LevelSchemes

logger = logging.getLogger(__name__)

isotope_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}
isotope_map_inverse = {v: k for k, v in isotope_map.items()}


class Isotope:
    """Represents a nuclear isotope with symbol, mass number, and atomic number.

    Args:
        isotope_name: Isotope identifier in format like 'Fe56', '56Fe', or '026056' (ZAA format).

    Raises:
        AssertionError: If the isotope symbol or mass number is invalid.

    Attributes:
        symbol: Element symbol (e.g., 'Fe', 'Au').
        A: Mass number (number of nucleons).
        Z: Atomic number (number of protons).
    """

    symbol: str
    A: int
    Z: int

    def __init__(self, isotope_name: str) -> None:
        self.symbol = "".join(re.findall(r"[A-Za-z]+", isotope_name))
        A_str: str = "".join(re.findall(r"\d+", isotope_name))

        if (len(self.symbol) == 0) and (len(isotope_name) == 6) and (len(A_str) == 6):
            self.Z = int(isotope_name[:3])
            self.A = int(isotope_name[3:])
            assert self.Z < self.A, (
                f"Invalid isotope: Z ({self.Z}) should not be smaller than A ({self.A})"
            )
            self.symbol = isotope_map_inverse[self.Z]

        else:
            assert 0 < len(self.symbol) <= 2, (
                f"Invalid isotope symbol '{self.symbol}': must be 1 or 2 letters."
            )
            z_val: int | None = isotope_map.get(self.symbol)
            if z_val is None:
                msg = f"Unknown isotope symbol: {self.symbol}"
                raise ValueError(msg)
            self.Z = z_val
            self.A = int(A_str)

        assert 1 <= self.A <= 300, (
            f"Invalid isotope mass '{self.A}': must be bw 1 and 300 amu."
        )

    def get_latex(self) -> str:
        """Return LaTeX representation of the isotope.

        Returns:
            str: LaTeX formatted isotope (e.g., $^{56}$Fe).
        """
        return f"$^{{{self.A}}}${self.symbol}"

    def get_talys_str(self) -> str:
        """Return TALYS-formatted isotope string (ZAA format).

        Returns:
            str: Zero-padded ZAA format string (e.g., '026056').
        """
        return f"{self.Z:03d}{self.A:03d}"

    def get_A_symbol(self) -> str:
        """Return isotope in A_symbol format.

        Returns:
            str: Mass number followed by symbol (e.g., '56Fe').
        """
        return f"{self.A}{self.symbol}"


def get_flux(file_path: str, graph_path: str) -> tuple[list[float], list[float]]:
    """Read neutron flux data from a ROOT file.

    Args:
        file_path: Path to the ROOT file containing the flux data.
        graph_path: Path to the TGraph within the ROOT file in format 'folder:graph_name'.
            Use '' for the top-level directory.

    Returns:
        tuple: Two lists (energies, flux values) extracted from the TGraph.

    Raises:
        OSError: If the ROOT file cannot be opened.
        ValueError: If the folder or graph is not found.
    """

    file = ROOT.TFile(file_path, "read")

    if not file.IsOpen():
        msg = f"Unable to open file: {file_path}"
        raise OSError(msg)

    folder, graph_name = graph_path.split(":")
    if folder:
        obj = file.Get(folder)
        if not obj:
            msg = f"Folder not found: {folder}"
            raise ValueError(msg)
    else:
        obj = file
    flux_graph = obj.Get(graph_name).Clone()

    if not flux_graph:
        msg = f"Graph not found: {graph_name} in {folder}"
        raise ValueError(msg)

    num_points = flux_graph.GetN()

    energies = []
    flux = []

    # Loop over the points and extract x and y values
    for i in range(num_points):
        x = flux_graph.GetX()[i]
        y = flux_graph.GetY()[i]
        energies.append(x)
        flux.append(y)

    return energies, flux


N_A = 6.023e23  # Avagadro's number


def calculate_common_range_flux_times_sigma(
    x_values: list[float],
    y_values: list[float],
    sigma_interp: np.ndarray,
    min_neutron_energy: float,
    max_neutron_energy: float,
    target_mass: float,
    target_A: float,
    beam_current: float,
    beam_to_target_distance: float,
) -> dict[str, float | list[float]]:
    """Calculate reaction rates and flux-related quantities over an energy range.

    Integrates flux, cross-section, and calculates the number of nuclear reactions
    based on beam current, target properties, and beam geometry.

    Args:
        x_values: Neutron energy values (keV).
        y_values: Neutron flux values (n/cm^2/s).
        sigma_interp: Interpolated cross-section values for each energy.
        min_neutron_energy: Minimum neutron energy for integration (keV).
        max_neutron_energy: Maximum neutron energy for integration (keV).
        target_mass: Mass of the target material (g).
        target_A: Mass number of the target isotope.
        beam_current: Beam current (µA).
        beam_to_target_distance: Distance from beam spot to target center (cm).

    Returns:
        dict: Dictionary containing:
            - Total_flux: Integrated neutron flux
            - Flux: Normalized flux per unit area
            - Number_of_nuclei: Number of target nuclei
            - Average_sigma: Average cross-section (mb)
            - N_reactions_per_second: Reaction rate
            - n_rectionsVSneutron_energy: Reactions per energy bin
            - n_rectionsVSneutron_energy_temp: Temporary reaction storage
    """
    flux_tot: float = 0.0
    sigma_tot: float = 0.0
    flux_times_sigma_tot: float = 0.0
    n_reactions: list[float] = []
    n_reactions_temp: list[float] = []

    num_points = len(x_values)

    for i in range(
        1, num_points
    ):  # Start from index 1 to avoid index out of range in x_values[i-1]
        if min_neutron_energy <= x_values[i] <= max_neutron_energy:
            flux_tot += y_values[i] * (x_values[i] - x_values[i - 1])
            sigma_tot += sigma_interp[i] * (x_values[i] - x_values[i - 1])
            n_reactions.append(
                y_values[i]
                * (x_values[i] - x_values[i - 1])
                * sigma_interp[i]
                * (N_A * target_mass / target_A)
                * 1e-3
                * 1e-24
                * (beam_current / beam_to_target_distance / beam_to_target_distance)
            )
            flux_times_sigma_tot += (
                y_values[i] * (x_values[i] - x_values[i - 1]) * sigma_interp[i]
            )

            n_reactions_temp.append(
                y_values[i]
                * sigma_interp[i]
                * (N_A * target_mass / target_A)
                * 1e-3
                * 1e-24
                * (beam_current / beam_to_target_distance / beam_to_target_distance)
            )

        else:
            n_reactions_temp.append(0)

    return {
        "Total_flux": flux_tot,
        "Flux": flux_tot
        * beam_current
        / beam_to_target_distance
        / beam_to_target_distance,
        "Number_of_nuclei": N_A * target_mass / target_A,
        "Average_sigma": 1e-3 * flux_times_sigma_tot / flux_tot,
        "N_reactions_per_second": (N_A * target_mass / target_A)
        * (flux_times_sigma_tot)
        * 1e-3
        * 1e-24
        * (beam_current / beam_to_target_distance / beam_to_target_distance),
        "n_rectionsVSneutron_energy": n_reactions,
        "n_rectionsVSneutron_energy_temp": n_reactions_temp,
    }


def save_simulated_counts_1d(
    talys_output_path: str,
    flux_file_path: str,
    flux_graph_path: str,
    min_neutron_energy: float,
    target_mass: float,
    target_A: float,
    acceptance: float,
    run_time: float,
    beam_current: float,
    abundance: float,
    beam_to_target_distance: float,
    output_path: str,
) -> pd.DataFrame:
    """Simulate 1D gamma-ray counts from TALYS output and neutron flux data.

    Reads TALYS nuclear reaction output files, combines with neutron flux data,
    and calculates expected gamma-ray counts for each reaction channel.

    Args:
        talys_output_path: Directory containing TALYS output files (gam02*.tot).
        flux_file_path: Path to ROOT file with neutron flux data.
        flux_graph_path: Path to TGraph in ROOT file ('folder:graph_name').
        min_neutron_energy: Minimum neutron energy threshold (keV).
        target_mass: Target mass (g).
        target_A: Target mass number.
        acceptance: Detector acceptance factor (0-1).
        run_time: Data acquisition time (seconds).
        beam_current: Beam current (µA).
        abundance: Abundance of the isotope in the target (%).
        beam_to_target_distance: Beam to target distance (cm).
        output_path: Path for output CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns: isotope, gamma, counts, startinglevel, endlevel.
    """
    file_list = Path(talys_output_path).glob("gam02*.tot")

    df_talys = pd.DataFrame(
        columns=["isotope", "gamma", "counts", "startinglevel", "endlevel"]
    )

    flux_energies, flux_values = get_flux(flux_file_path, flux_graph_path)

    i = 0
    for i, file in tqdm(enumerate(file_list)):
        df_cs = pd.read_csv(file, delimiter=r"\s+", comment="#", header=None)
        df_cs.columns = ["E", "cs"]
        interpolated_cs = np.interp(flux_energies, df_cs["E"], df_cs["cs"])
        max_energy = min(max(flux_energies), max(df_cs["E"]))
        results = calculate_common_range_flux_times_sigma(
            flux_energies,
            flux_values,
            interpolated_cs,
            min_neutron_energy=min_neutron_energy,
            max_neutron_energy=max_energy,
            target_mass=target_mass,
            target_A=target_A,
            beam_current=beam_current,
            beam_to_target_distance=beam_to_target_distance,
        )

        total_counts = (
            cast("float", results["N_reactions_per_second"])
            * run_time
            * acceptance
            * abundance
        )

        with Path.open(file, encoding="utf-8") as f:
            lines = f.readlines()
            starting_level = round(float(lines[16].split(" ")[-1]) * 1e3, 1)  # keV
            end_level = round(float(lines[21].split(" ")[-1]) * 1e3, 1)  # keV
            gamma_energy = round(float(lines[24].split(" ")[-1]) * 1e3, 1)  # keV

        df_talys.loc[i] = [
            Isotope(file.name.lstrip("gam").split("L")[0]).get_A_symbol(),
            gamma_energy,
            total_counts,
            starting_level,
            end_level,
        ]

    df_talys.to_csv(output_path)
    return df_talys


def EFFIT_RadWare(
    xy: tuple[float, float],
    A: float,
    B: float,
    C: float,
    D: float,
    E: float,
    F: float,
    G: float,
) -> float:
    """Efficiency function for gamma-ray detection (RadWare format).

    Implements a two-component efficiency model combining polynomial functions.

    Args:
        xy: Tuple of (x, y) where x is typically energy.
        A-G: Fitting parameters for the efficiency function.

    Returns:
        float: Calculated efficiency value.
    """
    x, y = xy
    result: float = 1 / (
        (A + B * x + C * x**2) ** (-G) + (D + E * y + F * y**2) ** (-G)
    ) ** (1 / G)
    return result


def get_gate_counts(
    df_talys: pd.DataFrame, isotopes: list[str], gate_energy: float, gate_width: float
) -> dict[str, float]:
    """Get counts within a gamma-ray energy gate for specified isotopes.

    Args:
        df_talys: DataFrame with simulated gamma-ray data.
        isotopes: List of isotope identifiers to search.
        gate_energy: Center of the energy gate (keV).
        gate_width: Width of the energy gate (keV).

    Returns:
        dict: Dictionary mapping isotope names to counts within the gate.
    """
    gate_counts_ = {}

    for isotope_ in isotopes:
        gate_counts = df_talys.loc[
            (
                df_talys["gamma"].between(
                    gate_energy - gate_width, gate_energy + gate_width
                )
            )
            & (df_talys["isotope"] == isotope_),
            "counts",
        ].to_numpy()

        gate_counts_[isotope_] = gate_counts[0] if len(gate_counts) > 0 else 0
    return gate_counts_


# Convert individual strings or mixed entries to floats, inserting 100 where blank
def fix_intensity_list(
    i_gamma_raw: list[float] | str, e_gamma_raw: list[float] | str
) -> list[float]:
    """Parse and fix gamma-ray intensity lists from nuclear data tables.

    Handles various formats including string representations, missing values,
    and special characters like '≤100'.

    Args:
        i_gamma_raw: Raw intensity values (can be list or string).
        e_gamma_raw: Raw energy values (can be list or string).

    Returns:
        list: Fixed intensity values as floats, with 100.0 for missing entries.
    """
    # Convert strings like "[20, , ≤100]" safely
    try:
        i_list = (
            ast.literal_eval(i_gamma_raw)
            if isinstance(i_gamma_raw, str)
            else i_gamma_raw
        )
        e_list = (
            ast.literal_eval(e_gamma_raw)
            if isinstance(e_gamma_raw, str)
            else e_gamma_raw
        )
    except (ValueError, SyntaxError):
        return []

    # Fallback if i_list is not a list
    if not isinstance(i_list, list):
        logger.error("Failed to parse intensity list")
        return []

    # Pad/fix list to match e_list length
    fixed_list = []
    for i in range(len(e_list)):
        if i < len(i_list):
            val = str(i_list[i]).strip()
            if val in {"", ","}:
                fixed_list.append(100.0)
            else:
                # Try to clean values like '≤100' etc.
                cleaned = re.sub(r"[^\d.\-]", "", val)
                try:
                    fixed_list.append(float(cleaned))
                except ValueError:
                    fixed_list.append(100.0)
        else:
            fixed_list.append(100.0)  # Fill if fewer entries than E(gamma)

    return fixed_list


# Calculate branching ratio
# Now calculate BR
def calculate_br(intensity_list: list[float]) -> list[float]:
    """Calculate normalized branching ratios from intensity values.

    Args:
        intensity_list: List of gamma-ray intensities.

    Returns:
        list: Normalized branching ratios (0-1), empty list if input is empty.
    """
    if not intensity_list:
        return []
    total = sum(intensity_list)
    return [i / total for i in intensity_list]


def get_BR_for_gamma(
    level_schemes: LevelSchemes, isotope: str, gate_energy: float
) -> float | None:
    """Retrieve branching ratio for a specific gamma-ray energy.

    Args:
        level_shemes: Dictionary of level scheme DataFrames indexed by isotope.
        isotope: Isotope identifier.
        gate_energy: Gamma-ray energy to look up (keV).

    Returns:
        float or None: Branching ratio if found, None otherwise.
    """
    df = level_schemes.dfs[isotope]

    for _, row in df.iterrows():
        energies = row["E(γ)(keV)"]  # noqa: RUF001
        BRs = row["BR"]

        if isinstance(energies, list) and gate_energy in energies:
            idx = energies.index(gate_energy)
            if isinstance(BRs, list) and idx < len(BRs):
                return float(BRs[idx])

    return None


def get_ref_list_infinite_order(
    level_schemes: LevelSchemes,
    gate_energy: float,
    gate_width: float,
    max_order: int = 5,
) -> tuple[dict[str, dict[float, list[list[float | str]]]], int]:
    """Build decay chains of arbitrary order from nuclear level schemes.

    Iteratively finds higher-order gamma-ray coincidences starting from a given
    gate energy, building decay chains up to specified order.

    Args:
        level_schemes: LevelSchemes object containing nuclear level data.
        gate_energy: Energy of the gate (keV).
        gate_width: Width of the energy gate (keV).
        max_order: Maximum coincidence order to compute (default: 5).

    Returns:
        tuple: (result dict, final order reached).
            result is a nested dict: {isotope: {energy: [[gamma, direction, text, level], ...]}}.
    """

    result: dict[str, dict[float, list[list[Any]]]] = {}

    # Apply the functions to fix columns
    for j in level_schemes.dfs.values():
        j["I(γ)"] = j.apply(  # noqa: RUF001
            lambda row: fix_intensity_list(row["I(γ)"], row["E(γ)(keV)"]),  # noqa: RUF001
            axis=1,
        )
        j["BR"] = j["I(γ)"].apply(calculate_br)  # noqa: RUF001
    # Start with first-order coincidences
    prev_order = level_schemes.get_first_order_coincidences(
        gate_energy=gate_energy, gate_width=gate_width * 2, return_levels=True
    )

    order = 1  # Track coincidence order
    while True:
        new_order: dict[str, dict[float, list[list[Any]]]] = {}
        found_new = False

        for isotope in prev_order:
            if isotope not in result:
                result[isotope] = {}

            next_coincidences = level_schemes.get_first_order_coincidences(
                gate_energy=gate_energy, gate_width=0, return_levels=True
            )

            found_new = (
                _process_isotope(
                    new_order, result, isotope, next_coincidences, prev_order
                )
                or found_new
            )

        if not found_new:
            break

        prev_order = new_order
        order += 1

        # Merge new_order results into result
        for isotope, value in new_order.items():
            if isotope not in result:
                result[isotope] = {}
            for energy in value:
                if energy not in result[isotope]:
                    result[isotope][energy] = []
                result[isotope][energy].extend(value[energy])

        if order > max_order:
            break

    return result, order


def _process_isotope(
    new_order: dict[str, dict[float, list[list[Any]]]],
    result: dict[str, dict[float, list[list[Any]]]],
    isotope: str,
    next_coincidences: dict[str, dict[float, list[Any]]],
    prev_order: dict[str, dict[float, list[list[Any]]]],
) -> bool:
    found_new = False
    for energy, transitions in prev_order[isotope].items():
        if energy not in result[isotope]:
            result[isotope][energy] = transitions

        for transition in transitions:
            gamma_energy, direction, text, _ = transition

            for next_transition in next_coincidences.get(isotope, {}).get(
                gamma_energy, []
            ):
                if _process_transition(
                    new_order,
                    isotope,
                    direction,
                    text,
                    next_transition,
                ):
                    found_new = True
    return found_new


def _process_transition(
    new_order: dict[str, dict[float, list[list[Any]]]],
    isotope: str,
    direction: Any,
    text: str,
    next_transition: tuple[Any, Any, str, Any],
) -> bool:
    next_gamma, next_direction, next_text, next_level = next_transition

    if next_direction != direction:
        return False

    new_order.setdefault(isotope, {})
    new_order[isotope].setdefault(next_gamma, [])

    pre_text = text.split(":")[1].split(",")[0]
    try:
        spin_text = next_text.split("(")[-1].split(",")[1].rstrip(")")
    except IndexError:
        spin_text = " "
    try:
        BF_ntext = float(next_text.split("(")[-1].split(",")[0]) / 100
    except (ValueError, IndexError):
        BF_ntext = 100 / 100

    try:
        BF_text = float(text.rsplit("(", maxsplit=1)[-1].split(",")[0]) / 100
    except (ValueError, IndexError):
        BF_text = 100 / 100

    new_text = f"{isotope}: {next_gamma} || {pre_text}, ({BF_ntext * BF_text * 100},{spin_text})"

    new_order[isotope][next_gamma].append(
        [next_gamma, next_direction, new_text, next_level]
    )

    return True


def get_results_inf(
    df_talys: pd.DataFrame,
    level_schemes: LevelSchemes,
    gate_energy: float,
    gate_width: float,
    acceptance: float,
    max_order: int,
    efficiency_function: Callable[[float], float],
) -> dict[float, list[str | float]]:
    """Calculate expected counts for cascade gamma-ray transitions.

    Computes expected peak counts for gamma-ray cascades of arbitrary order,
    including branching ratio corrections and detection efficiency.

    Args:
        df_talys: DataFrame with simulated 1D gamma-ray counts.
        level_schemes: LevelSchemes object with nuclear level data.
        gate_energy: Gate energy (keV).
        gate_width: Gate width (keV).
        acceptance: Detector acceptance (0-1).
        max_order: Maximum coincidence order to consider.
        efficiency_function: Function that returns efficiency for a given energy.

    Returns:
        dict: Mapping of gamma energy to [isotope, expected_counts].
    """
    counts_g_inf = []

    infinite_order, _ = get_ref_list_infinite_order(
        level_schemes=level_schemes,
        gate_energy=gate_energy,
        gate_width=gate_width,
        max_order=max_order,
    )

    for gates in infinite_order.values():
        for gate in gates:
            for g in gates[gate]:
                g0 = float(g[0])
                g2 = str(g[2])
                g3 = float(g[3])
                if (
                    len(df_talys.query(f"abs(gamma - {g0}) < {gate_width}")) > 0
                ):  # if you dont have counts for some that because it is above talys levels cause talys and nudat energy levels dont match in number
                    counts = df_talys.loc[
                        (df_talys["gamma"].between(g0 - gate_width, g0 + gate_width))
                        & (df_talys["isotope"] == g2.split(":", maxsplit=1)[0].strip())
                        & (
                            df_talys["startinglevel"].between(
                                g3 - gate_width, g3 + gate_width
                            )
                        ),
                        "counts",
                    ].to_numpy()
                    if len(counts) == 1:
                        counts_g_inf.append([g, counts])

    results_inf = {}

    for i in counts_g_inf:
        value = i[0][2]
        split_value = value.split(",")

        isotope_proj = split_value[0].split(":")[0]

        gamma_chain_str = split_value[0].split(":")[1]
        # Extract all gamma energies
        gamma_energies = [float(g.strip()) for g in gamma_chain_str.split("||")]
        # Get intermediate gammas (exclude first and last)
        intermediate_gammas = gamma_energies[1:]
        # print(intermediate_gammas)

        proj_gamma = i[0][0]

        branching_ratio = get_BR_for_gamma(
            level_schemes, isotope_proj, proj_gamma
        )  # This is the final level Branching ratio not intermediate
        BR_gate = get_BR_for_gamma(level_schemes, isotope_proj, gate_energy)

        # Get BRs for intermediate gammas
        intermediate_brs = []
        for gamma in intermediate_gammas:
            br = get_BR_for_gamma(level_schemes, isotope_proj, gamma)
            intermediate_brs.append(br)

        intermediate_brs_filtered: list[float] = [
            b for b in intermediate_brs if b is not None
        ]
        multiply_brs_intermediate = (
            math.prod(intermediate_brs_filtered) if intermediate_brs_filtered else 1.0
        )
        # print(multiply_brs_intermediate)

        counts_gamma = i[-1]

        counts_gammma = counts_gamma[0] if counts_gamma.size > 0 else 0

        gamma_energy = i[0][0]
        eff = efficiency_function(gamma_energy) * acceptance

        eff_gate = efficiency_function(gate_energy) * acceptance

        isotope_check = i[0][2].split(":")[0].strip()

        gate_counts_ = get_gate_counts(
            df_talys, list(level_schemes.dfs.keys()), gate_energy, gate_width
        )

        if i[0][1] == "up":
            peak_expected_counts: float = (
                counts_gammma
                * (BR_gate if BR_gate is not None else 0.0)
                * eff_gate
                * multiply_brs_intermediate
            )
        elif i[0][1] == "down":
            peak_expected_counts = (
                gate_counts_[isotope_check]
                * (branching_ratio if branching_ratio is not None else 0.0)
                * eff
                * multiply_brs_intermediate
            )
        else:
            peak_expected_counts = 0.0

        results_inf[gamma_energy] = [isotope_check, peak_expected_counts]
    return results_inf


def save_simulated_projection_spectrum(
    simulated_1d_counts_input_path: str,
    level_scheme: LevelSchemes,
    gate_energy: float,
    gate_width: float,
    acceptance: float,
    efficiency_function: Callable[[float], float],
    resolution_function: Callable[[float], float],
    max_order: int,
    output_path: str,
) -> None:
    """Generate simulated 2D projection spectra and save to ROOT file.

    Creates energy-gated projection spectra for each isotope, applying
    detector resolution and efficiency corrections.

    Args:
        simulated_1d_counts_input_path: Path to CSV with simulated 1D counts.
        level_scheme: LevelSchemes object with nuclear level data.
        gate_energy: Gate energy (keV).
        gate_width: Gate width (keV).
        acceptance: Detector acceptance (0-1).
        efficiency_function: Function returning efficiency at given energy.
        resolution_function: Function returning energy resolution (sigma) at given energy.
        max_order: Maximum coincidence order to consider.
        output_path: Path for output ROOT file.
    """

    df_talys = pd.read_csv(simulated_1d_counts_input_path, index_col=0)
    results_inf = get_results_inf(
        df_talys,
        level_scheme,
        gate_energy,
        gate_width,
        acceptance,
        max_order,
        efficiency_function,
    )
    # Define energy bins
    energy_bins = np.arange(0, 5000.5, 0.5)
    n_bins = len(energy_bins) - 1

    # Create ROOT file
    root_file = ROOT.TFile(output_path, "RECREATE")

    df_results_inf = pd.DataFrame(
        [
            {"Energy": energy, "Isotope": val[0], "Value": val[1]}
            for energy, val in results_inf.items()
        ]
    )

    df_results_inf.sort_values(by="Energy")

    for isotope, group in df_results_inf.groupby("Isotope"):
        # Create histogram for this isotope
        hist = ROOT.TH1F(
            f"{isotope}_spectrum",
            f"Spectrum for {isotope}",
            n_bins,
            energy_bins[0],
            energy_bins[-1],
        )
        spectrum = np.zeros(n_bins)

        for _, row in group.iterrows():
            gamma = row["Energy"]
            if gamma < energy_bins[0] or gamma > energy_bins[-1]:
                continue

            sigma_val = resolution_function(gamma)
            bin_centers = 0.5 * (energy_bins[1:] + energy_bins[:-1])
            pdf = norm.pdf(bin_centers, gamma, sigma_val)
            if row["Value"] == row["Value"]:  # Just to skip NaN values
                scale = (
                    efficiency_function(gamma)
                    * row["Value"]
                    / np.trapezoid(pdf, energy_bins)
                )
                spectrum += pdf * scale

        # Fill histogram
        for i in range(n_bins):
            hist.SetBinContent(i + 1, spectrum[i])

        hist.Write()

    root_file.Close()
