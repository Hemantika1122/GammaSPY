"""Module for 2D histogram operations.

This module provides a wrapper class for ROOT TH2 histograms with additional
manipulation methods, including symmetrization, projection, and visualization
functionalities for calibrated 2D histograms from ROOT files.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import ROOT  # pylint: disable=import-error
from ROOT import TH1, TH2, TCanvas  # pylint: disable=import-error
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class Hist2D:
    """A wrapper class for ROOT TH2 histograms with additional manipulation methods.

    This class provides a convenient interface for working with calibrated 2D
    histograms from ROOT files, with methods for common operations.

    Note:
        The histogram_path must be in the format "folder:histogram_name"
        (e.g., "Addback_gg:All Clovers").

    Parameters:
        file_path: Path to the ROOT file.
        histogram_path: Path to the histogram in the ROOT file, in the format
            "folder:histogram_name" (e.g., "Addback_gg_sym:All Clovers_sym").

    Attributes:
        histogram: The underlying ROOT TH2 object.
        file: The ROOT TFile object (kept open to maintain histogram access).
        bin_width: The bin width of underlying TH2 object (expect same for both axis)

    Raises:
        OSError: If the ROOT file cannot be opened.
        ValueError: If the folder, histogram is not found, or if bin widths
            are not uniform between X and Y axes.
        ValueError: If symmetrize is True and the histogram is not square.
    """

    def __init__(self, file_path: str, histogram_path: str, symmetrize: bool = True):
        """Initialize Hist2D by loading a histogram from a ROOT file.
        The 2D histogram is symmetrized if it is not symmetric already.

        This method sums the values in symmetric bins (i, j) and (j, i),
        storing the result in both bins. The histogram must be square
        (same number of bins in X and Y).

        Parameters:
            file_path: Path to the ROOT file.
            histogram_path: Path to the histogram in the ROOT file,
                in the format "folder:histogram_name".
            symmetrize: Boolean argument to enable symmetrizing the
                2D histogram (True by default)

        Raises:
            ValueError: If the histogram is not square.
        """
        self.file = ROOT.TFile(file_path, "read")

        if not self.file.IsOpen():
            msg = f"Unable to open file: {file_path}"
            raise OSError(msg)

        folder, hist_name = histogram_path.split(":")
        obj = self.file.Get(folder)
        if not obj:
            msg = f"Folder not found: {folder}"
            raise ValueError(msg)
        self.histogram = obj.Get(hist_name).Clone()

        if not self.histogram:
            msg = f"Histogram not found: {hist_name} in {folder}"
            raise ValueError(msg)

        bin_width_x = self.histogram.GetXaxis().GetBinWidth(1)
        bin_width_y = self.histogram.GetYaxis().GetBinWidth(1)
        if bin_width_x != bin_width_y:
            msg = f"Histogram is has non-uniform bin widths: ({bin_width_x} in x-axis and {bin_width_y} in y-axis)"
            raise ValueError(msg)
        self.bin_width = bin_width_y

        if symmetrize:
            self.symmetrize_histogram()

    def get_histogram(self) -> TH2:
        """Get the underlying ROOT TH2 object.

        This returns a clone of the histogram, so modifications to the
        returned object will not affect the internal histogram.

        Returns:
            The ROOT TH2 histogram object (a clone).
        """
        return self.histogram

    def symmetrize_histogram(self) -> None:
        """Symmetrize the underlying ROOT TH2 object if it is not symmetric already.

        This method sums the values in symmetric bins (i, j) and (j, i),
        storing the result in both bins. The histogram must be square
        (same number of bins in X and Y).

        Raises:
            ValueError: If the histogram is not square.

        Returns:
            None
        """
        n_bins_x = self.histogram.GetNbinsX()
        n_bins_y = self.histogram.GetNbinsY()

        if n_bins_x != n_bins_y:
            msg = (
                f"Cannot symmetrize: histogram is not square ({n_bins_x} x {n_bins_y})"
            )
            raise ValueError(msg)

        # Write core logic in C++ code in a string
        cpp_code_check_symmetry = """
        bool CheckSymmetryTH2(TH2* h, const int nx, const int ny)
        {
            bool is_symmetric = true;

            for (int ix = 1; ix <= nx; ++ix) {
                for (int iy = ix + 1; iy <= ny; ++iy) {
                    if (h->GetBinContent(ix, iy) != h->GetBinContent(iy, ix)){
                        is_symmetric=false;
                        break;
                    }
                }
                if (!is_symmetric){
                    break;
                }
            }
            return is_symmetric;
        }
        """
        # Inject the code in the ROOT interpreter
        ROOT.gInterpreter.ProcessLine(cpp_code_check_symmetry)
        is_symmetric = ROOT.CheckSymmetryTH2(self.histogram, n_bins_x, n_bins_y)

        if is_symmetric:
            logger.info("Histogram %s is already symmetric", self.histogram.GetName())
            return

        cpp_code_symmetrize = """
        void SymmetrizeTH2(TH2* h, const int nx, const int ny)
        {
            for (int i = 1; i <= nx; ++i) {
                for (int j = i + 1; j <= ny; ++j) {
                    auto content_ij = h->GetBinContent(i, j);
                    auto content_ji = h->GetBinContent(j, i);
                    auto sym_content = content_ij + content_ji;

                    h->SetBinContent(i, j, sym_content);
                    h->SetBinContent(j, i, sym_content);
                }
            }
            return;
        }
        """
        # Inject the code in the ROOT interpreter
        ROOT.gInterpreter.ProcessLine(cpp_code_symmetrize)
        ROOT.SymmetrizeTH2(self.histogram, n_bins_x, n_bins_y)

        return

    def get_gate_bins(self, gate_energy: float, gate_width: float) -> list[int]:
        """Calculate the bin range for a gate around the given energy.

        Converts the gate energy and width to bin indices, ensuring bounds
        are within the histogram range.

        Parameters:
            gate_energy: Center energy for the gate.
            gate_width: Total width of the gate.

        Returns:
            A list containing [first_bin, last_bin] for the gate range.
        """
        gate = (
            round(gate_energy) - gate_width / 2,
            round(gate_energy) + gate_width / 2,
        )
        gate_bins = [
            math.floor(gate[0] / self.bin_width),
            math.ceil(gate[1] / self.bin_width),
        ]
        gate_bins[0] = max(gate_bins[0], 1)
        gate_bins[1] = min(gate_bins[1], self.histogram.GetNbinsY())

        return gate_bins

    def get_projection(
        self,
        gate_energy: float,
        gate_width: float,
        unit: str = "keV",
    ) -> TH1:
        """Apply a gate and return the projected 1D spectrum.

        Applies a gate around gate_energy ± gate_width/2 and projects
        the 2D histogram onto the X-axis to produce a 1D spectrum.

        Parameters:
            gate_energy: Center energy for the gate.
            gate_width: Total width of the gate.
            unit: Energy units (default: keV).

        Returns:
            The projected ROOT TH1 histogram object.
        """

        logger.info(
            "Using fixed gate energy %f instead of looking in the database", gate_energy
        )

        gate_bins = self.get_gate_bins(gate_energy=gate_energy, gate_width=gate_width)

        pro = self.histogram.ProjectionX(
            name=f"Gated in [{gate_bins[0] * self.bin_width} - {gate_bins[1] * self.bin_width}] {unit}",
            firstybin=gate_bins[0],
            lastybin=gate_bins[1],
        )

        return pro.Clone()

    def get_2d_background(self, gate_energy: float, gate_width: float) -> TH1:
        """Calculate the background contribution within the gate region.

        Estimates the background by computing the total projection, subtracting
        a background estimate via ShowBackground, and normalizing to the gate region.

        Parameters:
            gate_energy: Center energy for the gate.
            gate_width: Total width of the gate.

        Returns:
            The background histogram for the gated region.
        """
        n_bins = self.histogram.GetNbinsX()
        total_projection = self.get_projection(
            gate_energy=0.5 * n_bins * self.bin_width,  # Central bin
            gate_width=2 * n_bins * self.bin_width,  # full range
        )
        total_background = total_projection.ShowBackground(20)

        total_signal = total_projection.Clone()
        total_signal.Add(total_background, -1)

        total_yield = total_projection.Integral()

        p_norm = 0
        b_norm = 0

        gate_bins = self.get_gate_bins(gate_energy=gate_energy, gate_width=gate_width)
        for j in range(gate_bins[0], gate_bins[1] + 1):
            p_norm += total_signal.GetBinContent(j)
            b_norm += total_background.GetBinContent(j)

        pi = total_projection.Clone()
        bi = total_background.Clone()

        pi.Scale(b_norm / total_yield)
        bi.Scale(p_norm / total_yield)

        bkg_gate = pi.Clone()
        bkg_gate.Add(bi, 1)

        return bkg_gate

    def draw_projection(
        self,
        gate_energy: float,
        gate_width: float,
        unit: str = "keV",
        show_title: bool = False,
        show_stats: bool = True,
        subtract_background: bool = True,
        sideband_background_gate_energies: list[float] | None = None,
    ) -> TCanvas:
        """Apply a gate, project the spectrum, and draw it on a canvas.

        Applies a gate around gate_energy ± gate_width/2, projects the 2D
        histogram onto the X-axis, and draws the result on a ROOT TCanvas.

        Parameters:
            gate_energy: Center energy for the gate.
            gate_width: Total width of the gate.
            unit: Energy units (default: keV).
            show_title: Whether to show the histogram title (default: False).
            show_stats: Whether to show statistics box (default: True).
            subtract_background: Whether to subtract background (default: True).
            sideband_background_gate_energies: List of gate energies to be used for background subtraction. If none provided, 2d background subtraction method is used by default.

        Returns:
            The ROOT TCanvas with the projected histogram.
        """

        projected_histogram = self.get_projection(
            gate_energy=gate_energy, gate_width=gate_width, unit=unit
        )

        if subtract_background:
            if sideband_background_gate_energies is None:
                projected_background = self.get_2d_background(
                    gate_energy=gate_energy, gate_width=gate_width
                )
                projected_histogram.Add(projected_background, -1)
            else:
                pro_sb = {}
                for i, sideband_gate_energy in enumerate(
                    sideband_background_gate_energies
                ):
                    pro_sb[i] = self.get_projection(
                        gate_energy=sideband_gate_energy, gate_width=gate_width
                    )
                    pro_sb[i].Scale(1 / len(sideband_background_gate_energies))
                    projected_histogram.Add(pro_sb[i], -1)

        canvas = TCanvas("Gated", "Gated", 1450, 1000)
        projected_histogram.GetYaxis().SetTitle(f"Counts/{self.bin_width} {unit}")
        projected_histogram.GetYaxis().SetTitleFont(
            42
        )  # 42 is the code for Times New Roman font
        projected_histogram.GetYaxis().SetTitleSize(0.04)
        projected_histogram.GetXaxis().SetTitle("E_{#gamma}")
        projected_histogram.GetXaxis().SetTitleFont(
            42
        )  # 42 is the code for Times New Roman font
        projected_histogram.GetXaxis().SetTitleSize(0.04)

        if not show_stats:
            projected_histogram.SetStats(0)
        if not show_title:
            projected_histogram.SetTitle("")

        projected_histogram.Draw("HISTSAME")

        canvas.Draw()

        return canvas

    def close(self) -> None:
        """Close the ROOT file and release resources."""
        if self.file and self.file.IsOpen():
            self.file.Close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Context manager exit - ensures file is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures file is closed on object destruction."""
        self.close()
