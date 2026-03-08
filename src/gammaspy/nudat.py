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

    def get_first_order_coincidences(
        self,
        gate_energy: float,
        gate_width: float,
        return_levels: bool = False,
        require_starting_level: float | None = None,
    ) -> dict[str, dict[float, list[Any]]]:
        """Find first-order gamma-ray coincidences for a given gate energy.

        Searches for gamma rays that are in coincidence with a gating gamma
        by finding transitions that connect to either the starting or ending
        level of the gate gamma.

        Parameters
        ----------
        gate_energy : float
            Center energy of the gate in keV.
        gate_width : float
            Width of the energy gate in keV (total width, not half-width).
        return_levels : bool, optional
            If True, include the starting level energy in the result.
            Defaults to False.
        require_starting_level : float | None, optional
            If provided, only include gates starting from this level energy.
            Defaults to None.

        Returns
        -------
        dict[str, dict[float, list[Any]]]
            Dictionary mapping isotope symbols to dictionaries of gate energies
            and their coincident gamma rays. Each coincident gamma is represented
            as a list containing: [energy, direction, description, level(optional)]
            where direction is "down" for de-excitation or "up" for excitation.

        Examples
        --------
        >>> schemes = LevelSchemes(["56Fe"], path=Path("./data"))
        >>> coincidences = schemes.get_coincidences(1234.5, 10.0)
        """
        result: dict[str, dict[float, list[Any]]] = {}

        for isotope, df in self.dfs.items():
            result[isotope] = {}

            for df_index, gamma_series in enumerate(df["E(γ)(keV)"]):  # noqa: RUF001
                if gamma_series is None:
                    continue

                for gamma_index, gamma_value in enumerate(gamma_series):
                    if abs(gamma_value - gate_energy) > gate_width / 2:
                        continue

                    gate_starting_level = df["E(level)(keV)"][df_index]
                    gate_ending_level = df["Final Levels"][df_index][gamma_index]

                    if (
                        require_starting_level is not None
                        and gate_starting_level != require_starting_level
                    ):
                        continue

                    result[isotope][gamma_value] = []

                    self._find_coincidences_in_df(
                        df,
                        gate_starting_level,
                        gate_ending_level,
                        result[isotope][gamma_value],
                        return_levels,
                    )

        return result

    def _find_coincidences_in_df(
        self,
        df: pd.DataFrame,
        gate_starting_level: float,
        gate_ending_level: float,
        coincidences: list[Any],
        return_levels: bool,
    ) -> None:
        """Find coincidences in a dataframe for given gate levels."""
        for coin_df_index in range(len(df)):
            coin_starting_level = df["E(level)(keV)"][coin_df_index]
            coin_gamma_series = df["E(γ)(keV)"][coin_df_index]  # noqa: RUF001
            coin_ending_levels = df["Final Levels"][coin_df_index]

            if coin_gamma_series is None or coin_ending_levels is None:
                continue

            for i, coin_ending_level in enumerate(coin_ending_levels):
                coin_gamma_energy = coin_gamma_series[i]

                if coin_starting_level == gate_ending_level:
                    direction = "down"
                    tt = self._build_transition_string(
                        df, coin_df_index, i, coin_gamma_energy
                    )
                    self._add_coincidence(
                        coincidences,
                        coin_gamma_energy,
                        direction,
                        tt,
                        return_levels,
                        coin_starting_level,
                    )

                if coin_ending_level == gate_starting_level:
                    direction = "up"
                    tt = self._build_transition_string(
                        df,
                        coin_df_index,
                        i,
                        coin_gamma_energy,
                        include_initial=True,
                    )
                    self._add_coincidence(
                        coincidences,
                        coin_gamma_energy,
                        direction,
                        tt,
                        return_levels,
                        coin_starting_level,
                    )

    def get_coincidence_paths(
        self,
        gate_energy: float,
        gate_width: float,
        max_order: int = 2,
        current_order: int = 1,
        starting_level: float | None = None,
        path_prefix: list[str] | None = None,
        required_direction: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively traverse the level scheme to find coincidence chains.

        This method extends the first-order coincidences by recursively
        following gamma transitions through the nuclear level scheme,
        building chains of coincident gammas up to a specified order.

        Parameters:
            gate_energy: Center energy of the initial gate.
            gate_width: Width of the energy gate.
            max_order: Maximum recursion depth/coincidence order (default: 2).
            current_order: Current recursion depth (used internally).
            starting_level: Starting level energy for this recursion step (used internally).
            path_prefix: List of gate energies in the current path (used internally).
            required_direction: Direction constraint for this step: "up" or "down" (used internally).

        Returns:
            List of dictionaries, each containing:
                - isotope: Isotope symbol.
                - energy: Gamma-ray energy in keV.
                - direction: Transition direction ("up" or "down").
                - label: Formatted transition description.
                - order: Coincidence order (1 = first-order, 2 = second-order, etc.).
                - chain: String representation of the coincidence path.

        Examples:
            >>> schemes = LevelSchemes(["56Fe"], path=Path("./data"))
            >>> paths = schemes.get_coincidence_paths(1234.5, 10.0, max_order=2)
            >>> for p in paths:
            ...     print(f"{p['isotope']}: {p['energy']} keV ({p['direction']}, order {p['order']})")
        """
        results = []
        if path_prefix is None:
            path_prefix = []

        # Get coincidences for this specific step
        coinc_data = self.get_first_order_coincidences(
            gate_energy=gate_energy,
            gate_width=gate_width * 2,
            require_starting_level=starting_level,
            return_levels=True,
        )

        for isotope, gates in coinc_data.items():
            for gate_found, coinc_list in gates.items():
                for coinc in coinc_list:
                    # coinc = [energy, direction, label, level_id, ...]
                    energy, direction, label, level_id = (
                        coinc[0],
                        coinc[1],
                        coinc[2],
                        coinc[3],
                    )
                    label.replace(f"{isotope}: ", "")

                    # Direction check: must match the flow of the chain
                    if (
                        required_direction is not None
                        and direction != required_direction
                    ):
                        continue

                    # Record this stage
                    current_path = [*path_prefix, str(gate_found)]
                    entry = {
                        "isotope": isotope,
                        "gate_found": gate_found,
                        "energy": energy,
                        "direction": direction,
                        "label": label,
                        "order": current_order,
                        "chain": " || ".join(current_path),
                    }
                    results.append(entry)

                    # Recursion: if we haven't hit max depth, keep going
                    if current_order < max_order:
                        sub_results = self.get_coincidence_paths(
                            gate_energy=energy,
                            gate_width=0,
                            max_order=max_order,
                            current_order=current_order + 1,
                            starting_level=level_id,
                            path_prefix=current_path,
                            required_direction=direction,
                        )
                        results.extend(sub_results)

        return results

    def _build_transition_string(
        self,
        df: pd.DataFrame,
        df_index: int,
        gamma_index: int,
        energy: float,
        include_initial: bool = False,
    ) -> str:
        """Build a formatted transition description string.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the nuclear data.
        isotope : str
            Isotope symbol.
        df_index : int
            DataFrame row index.
        gamma_index : int
            Index within the gamma series.
        energy : float
            Gamma-ray energy.
        include_initial : bool, optional
            If True, include the initial Jπ in the string. Defaults to False.

        Returns
        -------
        str
            Formatted transition description.
        """
        initial_jpi = ""
        final_jpi = ""
        intensity = ""
        multipolarity = ""

        try:
            initial_jpi = f"{df['Jπ(level)'][df_index].replace('+', '^{+}')} -> "
            final_jpi = f"{df['Final Jπ'][df_index][gamma_index].replace('+', '^{+}')}"
            intensity = f"({df['I(γ)'][df_index][gamma_index].replace('≤', '#leq')}"  # noqa: RUF001
            multipolarity = f", {df['M(γ)'][df_index][gamma_index]})"  # noqa: RUF001
        except (IndexError, TypeError, AttributeError):
            try:
                final_jpi = (
                    f"-> {df['Final Jπ'][df_index][gamma_index].replace('+', '^{+}')}"
                )
                intensity = (
                    f"({df['I(γ)'][df_index][gamma_index].replace('≤', '#leq')})"  # noqa: RUF001
                )
            except (IndexError, TypeError, AttributeError):
                pass

        if include_initial and initial_jpi:
            return f"{energy}, {initial_jpi}{final_jpi} {intensity}{multipolarity}"
        return f"{energy}, {final_jpi} {intensity}{multipolarity}"

    def _add_coincidence(
        self,
        coincidences: list[Any],
        energy: float,
        direction: str,
        description: str,
        return_levels: bool,
        level: float,
    ) -> None:
        """Add a coincidence entry to the result list.

        Parameters
        ----------
        coincidences : list[Any]
            List to append the coincidence to.
        energy : float
            Gamma-ray energy.
        direction : str
            Direction of transition ("up" or "down").
        description : str
            Formatted description of the transition.
        return_levels : bool
            Whether to include level information.
        level : float
            Starting level energy.
        """
        if return_levels:
            coincidences.append([energy, direction, description, level])
        else:
            coincidences.append([energy, direction, description])


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
