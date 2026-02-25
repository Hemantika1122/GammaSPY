"""Module for 2D histogram operations."""

from __future__ import annotations

import logging

from ROOT import TH2D, TFile

logger = logging.getLogger(__name__)


class Hist2D:
    """A wrapper class for ROOT TH2D histograms with additional manipulation methods.

    This class provides a convenient interface for working with calibrated 2D
    histograms from ROOT files, with methods for common operations.

    Parameters:
        file_path: Path to the ROOT file.
        histogram_path: Path to the histogram in the ROOT file, in the format
            "folder:histogram_name" (e.g., "Addback_gg_sym:All Clovers_sym").

    Attributes:
        histogram: The underlying ROOT TH2D object.
        file: The ROOT TFile object (kept open to maintain histogram access).
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
        self.file = TFile(file_path, "read")

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

        if symmetrize:
            self.symmetrize_histogram()

    def get_histogram(self) -> TH2D:
        """Get the underlying ROOT TH2D object.

        Returns:
            The ROOT TH2D histogram object.
        """
        return self.histogram

    def symmetrize_histogram(self):
        """Symmetrize the underlying ROOT TH2D object if it is not symmetric already.

        This method sums the values in symmetric bins (i, j) and (j, i),
        storing the result in both bins. The histogram must be square
        (same number of bins in X and Y).

        Returns:
            The ROOT TH2D histogram object.
        """
        n_bins_x = self.histogram.GetNbinsX()
        n_bins_y = self.histogram.GetNbinsY()

        if n_bins_x != n_bins_y:
            msg = (
                f"Cannot symmetrize: histogram is not square ({n_bins_x} x {n_bins_y})"
            )
            raise ValueError(msg)

        is_symmetric = True
        for i in range(1, n_bins_x + 1):
            for j in range(i + 1, n_bins_y + 1):
                if self.histogram.GetBinContent(
                    i, j
                ) != self.histogram.GetBinContent(j, i):
                    is_symmetric = False
                    break
            if not is_symmetric:
                break

        if is_symmetric:
            logger.info(
                "Histogram %s is already symmetric", self.histogram.GetName()
            )
            return

        for i in range(1, n_bins_x + 1):
            for j in range(i + 1, n_bins_y + 1):
                content_ij = self.histogram.GetBinContent(i, j)
                content_ji = self.histogram.GetBinContent(j, i)
                sym_content = content_ij + content_ji

                self.histogram.SetBinContent(i, j, sym_content)
                self.histogram.SetBinContent(j, i, sym_content)

        return

    def close(self) -> None:
        """Close the ROOT file and release resources."""
        if self.file and self.file.IsOpen():
            self.file.Close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is closed."""
        self.close()

    def __del__(self):
        """Destructor - ensures file is closed on object destruction."""
        self.close()
