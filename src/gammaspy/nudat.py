from pathlib import Path

import pandas as pd


class LevelSchemes:
    def __init__(self, isotopes: list[str], path: Path | None = None) -> None:
        if path is None:
            path = Path.cwd()

        self.df = {}
        self.Egammas = {}

        for d in isotopes:
            self.df[d] = self.read_database(
                filename=path / f"adoptedLevels{d}.csv", head=None
            )  # None reads entire databases

            self.Egammas[d] = [i for i in self.df[d]["E(γ)(keV)"] if i is not None]
            self.Egammas[d] = [item for i in self.Egammas[d] for item in i]

    def read_database(self, filename, head=20):
        database = pd.read_csv(filename, index_col=False, usecols=range(9))

        if head is None:
            head = database[database["E(γ)(keV)"] == "Multipolarity"].index[0]

        database = database.head(head)

        database.drop(["XREF", "T1/2(level)"], axis=1, inplace=True)
        database.rename({"Unnamed: 8": "Final Jπ"}, axis=1, inplace=True)

        database["E(γ)(keV)"] = database["E(γ)(keV)"].apply(nudata_energy_cleanup)
        database["Final Levels"] = database["Final Levels"].apply(nudata_energy_cleanup)
        database["Final Jπ"] = database["Final Jπ"].apply(nudata_state_cleanup)
        database["M(γ)"] = database["M(γ)"].apply(nudata_state_cleanup)
        database["I(γ)"] = database["I(γ)"].apply(nudata_state_cleanup)
        database["E(level)(keV)"] = database["E(level)(keV)"].apply(
            lambda x: nudata_energy_cleanup(x)[0] if isinstance(x, str) else x
        )

        return database

    def get_database(self, isotope: str):
        return self.df[isotope]


def nudata_energy_cleanup(row):
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
        cleaned = []
        for i in temp:
            try:
                cleaned.append(float(i))
            except ValueError:
                cleaned.append(None)  # Use None for non-convertible entries
        return cleaned
    return []


def nudata_state_cleanup(row):
    if isinstance(row, str):
        return [
            i.lstrip().rstrip().split(" ")[0].replace("?", "") for i in row.split(",")
        ]
    return []
