"""Module for reading and processing nuclear data from the NuDat database.

This module provides functionality to read gamma-ray spectroscopy data from
NuDat (National Nuclear Data Center) CSV files and process them into
usable formats.

References
----------
.. [1] National Nuclear Data Center, NuDat 3.0.
       https://www.nndc.bnl.gov/nudat3/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class LevelSchemes:
    """Class to handle nuclear level scheme data from NuDat.

    This class reads and processes gamma-ray spectroscopy data from
    NuDat database CSV files for specified isotopes.

    Parameters
    ----------
    isotopes : list[str]
        List of isotope symbols (e.g., ["56Fe", "60Ni"]).
    path : Path, optional
        Directory containing the NuDat CSV files. Defaults to current
        working directory.

    Attributes
    ----------
    df : dict[str, pd.DataFrame]
        Dictionary mapping isotope symbols to their DataFrames.
    gamma_energies : dict[str, list[float]]
        Dictionary mapping isotope symbols to lists of gamma-ray energies
        in keV.

    Examples
    --------
    >>> schemes = LevelSchemes(["56Fe"], path=Path("./data"))
    >>> fe_data = schemes.get_database("56Fe")
    >>> fe_gammas = schemes.gamma_energies["56Fe"]
    """

    def __init__(self, isotopes: list[str], path: Path | None = None) -> None:
        """Initialize the LevelSchemes object."""
        if path is None:
            path = Path.cwd()

        self.dfs: dict[str, pd.DataFrame] = {}
        self.gamma_energies: dict[str, list[float | None]] = {}

        for isotope in isotopes:
            self.dfs[isotope] = self._read_database(
                filename=path / f"adoptedLevels{isotope}.csv",
                head=None,
            )
            gammas = self.dfs[isotope]["E(γ)(keV)"]  # noqa: RUF001
            self.gamma_energies[isotope] = [
                item for val in gammas if val is not None for item in val
            ]

    def _read_database(self, filename: Path, head: int | None = None) -> pd.DataFrame:
        """Read a NuDat CSV database file.

        Parameters
        ----------
        filename : Path
            Path to the CSV file.
        head : int | None, optional
            Number of rows to read. If None, reads up to the Multipolarity
            row (end of data).

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with cleaned column names and data types.
        """
        database = pd.read_csv(filename, index_col=False, usecols=range(9))

        if head is None:
            matches = database[database["E(γ)(keV)"] == "Multipolarity"]  # noqa: RUF001
            if not matches.empty:
                head = int(matches.index.tolist()[0])
            else:
                head = len(database)

        database = database.head(head)

        database = database.drop(["XREF", "T1/2(level)"], axis=1)
        database = database.rename(columns={"Unnamed: 8": "Final Jπ"})

        database["E(γ)(keV)"] = database["E(γ)(keV)"].apply(nudata_energy_cleanup)  # noqa: RUF001
        database["Final Levels"] = database["Final Levels"].apply(nudata_energy_cleanup)
        database["Final Jπ"] = database["Final Jπ"].apply(nudata_state_cleanup)
        database["M(γ)"] = database["M(γ)"].apply(nudata_state_cleanup)  # noqa: RUF001
        database["I(γ)"] = database["I(γ)"].apply(nudata_state_cleanup)  # noqa: RUF001
        database["E(level)(keV)"] = database["E(level)(keV)"].apply(
            lambda x: nudata_energy_cleanup(x)[0] if isinstance(x, str) else x
        )

        return database

    def get_database(self, isotope: str) -> pd.DataFrame:
        """Get the DataFrame for a specific isotope.

        Parameters
        ----------
        isotope : str
            Isotope symbol (e.g., "56Fe").

        Returns
        -------
        pd.DataFrame
            DataFrame containing the level scheme data.

        Raises
        ------
        KeyError
            If the isotope is not in the loaded data.
        """
        return self.dfs[isotope]

    def get_coincidences(
        self,
        gate_energy: float,
        gate_width: float,
        return_levels: bool = False,
        require_starting_level: float | None = None,
    ) -> dict[str, list[Any]]:

        result = {}

        for isotope, df in self.dfs.items():
            result[isotope] = {}
            for df_index, gamma_series in enumerate(df["E(γ)(keV)"]):
                if gamma_series is not None:  # Checking for non-null values
                    for gamma_index, gamma_value in enumerate(gamma_series):
                        if abs(gamma_value - gate_energy) <= gate_width / 2:
                            gate_starting_level = df["E(level)(keV)"][df_index]
                            gate_ending_level = df["Final Levels"][df_index][
                                gamma_index
                            ]  # Ending level energy of the gate gamma

                            if (
                                require_starting_level
                                and gate_starting_level != require_starting_level
                            ):
                                continue

                            result[isotope][gamma_value] = []

                            for coin_df_index, coin_starting_level in enumerate(
                                df["E(level)(keV)"]
                            ):
                                if (coin_starting_level == gate_ending_level) and (
                                    df["E(γ)(keV)"][coin_df_index] is not None
                                ):
                                    for i, coin_gamma_energy in enumerate(
                                        df["E(γ)(keV)"][coin_df_index]
                                    ):
                                        tt = f"{isotope}: {coin_gamma_energy}"
                                        try:
                                            tt += f", {df['Jπ(level)'][coin_df_index].replace('+', '^{+}')} -> {df['Final Jπ'][coin_df_index][i].replace('+', '^{+}')} ({df['I(γ)'][coin_df_index][i].replace('≤', '#leq')}, {df['M(γ)'][coin_df_index][i]})"
                                        except:
                                            tt += f", ({df['I(γ)'][coin_df_index][i].replace('≤', '#leq')})"

                                        if return_levels:
                                            result[isotope][gamma_value].append(
                                                [coin_gamma_energy, "down", tt, coin_starting_level]
                                            )
                                        else:
                                            result[isotope][gamma_value].append(
                                                [coin_gamma_energy, "down", tt]
                                            )

                            for coin_df_index, coin_ending_level_series in enumerate(
                                df["Final Levels"]
                            ):
                                if coin_ending_level_series is not None:
                                    for i, j in enumerate(coin_ending_level_series):
                                        if j == gate_starting_level:
                                            coin_gamma_energy = df["E(γ)(keV)"][
                                                coin_df_index
                                            ][i]
                                            g_up_startinglevel = df["E(level)(keV)"][
                                                coin_df_index
                                            ]

                                            tt = f"{isotope}: {coin_gamma_energy}"
                                            try:
                                                tt += f", {df['Jπ(level)'][coin_df_index].replace('+', '^{+}')} -> {df['Final Jπ'][coin_df_index][i].replace('+', '^{+}')} ({df['I(γ)'][coin_df_index][i].replace('≤', '#leq')}, {df['M(γ)'][coin_df_index][i]})"
                                            except:
                                                tt += f", -> {df['Final Jπ'][coin_df_index][i].replace('+', '^{+}')} ({df['I(γ)'][coin_df_index][i].replace('≤', '#leq')})"

                                            if return_levels:
                                                result[isotope][gamma_value].append(
                                                    [
                                                        coin_gamma_energy,
                                                        "up",
                                                        tt,
                                                        g_up_startinglevel,
                                                    ]
                                                )
                                            else:
                                                result[isotope][gamma_value].append(
                                                    [coin_gamma_energy, "up", tt]
                                                )

        return result


def nudata_energy_cleanup(row: Any) -> list[float | None]:
    """Clean and parse energy values from NuDat format.

    Handles various formatting cases including uncertainties, limits,
    and multiple values in a single cell.

    Parameters
    ----------
    row : Any
        Energy value(s) from NuDat. Can be a string with comma-separated
        values, numeric types, or None.

    Returns
    -------
    list[float | None]
        List of cleaned energy values. Returns empty list for None or
        non-string inputs. Values that cannot be converted to float
        are returned as None.

    Examples
    --------
    >>> nudata_energy_cleanup("1234.5, 5678.9")
    [1234.5, 5678.9]
    >>> nudata_energy_cleanup("≥100.0")
    [100.0]
    >>> nudata_energy_cleanup("?")
    [None]
    """
    if isinstance(row, str):
        temp = [
            i.lstrip()
            .rstrip()
            .split(" ")[0]
            .replace("?", "")
            .replace("≥", "")
            .lstrip("≈")
            .rstrip("S")
            .rstrip("X")
            .rstrip("+")
            for i in row.split(",")
        ]

        def _try_float(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None

        return [_try_float(i) for i in temp]
    return []


def nudata_state_cleanup(row: Any) -> list[str]:
    """Clean and parse nuclear state quantum numbers from NuDat format.

    Parses spin/parity values and other state identifiers from NuDat
    formatted strings.

    Parameters
    ----------
    row : Any
        State value(s) from NuDat. Can be a string with comma-separated
        quantum numbers, or other types.

    Returns
    -------
    list[str]
        List of cleaned state values. Returns empty list for None or
        non-string inputs.

    Examples
    --------
    >>> nudata_state_cleanup("2+, 3+")
    ['2+', '3+']
    >>> nudata_state_cleanup("1/2-?")
    ['1/2-']
    """
    if isinstance(row, str):
        return [
            i.lstrip().rstrip().split(" ")[0].replace("?", "") for i in row.split(",")
        ]
    return []
