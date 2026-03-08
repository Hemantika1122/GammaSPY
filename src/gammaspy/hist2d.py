"""Module for 2D histogram operations."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import ROOT  # pylint: disable=import-error
from ROOT import TH1, TH2  # pylint: disable=import-error
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class Hist2D:
    """A wrapper class for ROOT TH2 histograms with additional manipulation methods.

    This class provides a convenient interface for working with calibrated 2D
    histograms from ROOT files, with methods for common operations.

    Parameters:
        file_path: Path to the ROOT file.
        histogram_path: Path to the histogram in the ROOT file, in the format
            "folder:histogram_name" (e.g., "Addback_gg_sym:All Clovers_sym").

    Attributes:
        histogram: The underlying ROOT TH2 object.
        file: The ROOT TFile object (kept open to maintain histogram access).
        bin_width: The bin width of underlying TH2 object (expect same for both axis)
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
            msg = (
                f"Histogram is has non-uniform bin widths: ({bin_width_x} in x-axis and {bin_width_y} in y-axis)"
            )
            raise ValueError(msg)
        self.bin_width = bin_width_y
        
        if symmetrize:
            self.symmetrize_histogram()

    def get_histogram(self) -> TH2:
        """Get the underlying ROOT TH2 object.

        Returns:
            The ROOT TH2 histogram object.
        """
        return self.histogram

    def symmetrize_histogram(self) -> None:
        """Symmetrize the underlying ROOT TH2 object if it is not symmetric already.

        This method sums the values in symmetric bins (i, j) and (j, i),
        storing the result in both bins. The histogram must be square
        (same number of bins in X and Y).

        Returns:
            The ROOT TH2 histogram object.
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

    def get_projection(self, gate_energy: float, gate_width: float, unit:str = "keV") -> TH1:
        """Applies a gate of gate_energy ± gate_width/2 and returns the projected spectrum.
            
        Parameters:
            gate_energy: Gate energy
            gate_width: Gate width
            unit: Energy units (default: keV)
        Returns:
            The projected ROOT TH1 histogram object.
        """

        logger.info("Using fixed gate energy %f instead of looking in the database", gate_energy)
            
        gate = (round(gate_energy)-gate_width/2,round(gate_energy)+gate_width/2)
        gate_bin = [math.floor(gate[0]/self.bin_width), math.ceil(gate[1]/self.bin_width)]
        gate_bin[0] = max(gate_bin[0], 1)
        gate_bin[1] = min(gate_bin[1], self.histogram.GetNbinsY())
    
        pro_X = self.histogram.ProjectionX(name = f"Gated in [{gate_bin[0]*self.bin_width} - {gate_bin[1]*self.bin_width}] {unit}",
                                           firstybin=gate_bin[0],
                                           lastybin=gate_bin[1])

        return pro_X.Clone()

        
        
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
