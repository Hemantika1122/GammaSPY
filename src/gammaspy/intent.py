try:
    from ROOT import TFile
except ImportError:
    print(f"Looks like root is not installed.")

def open_output(file_path):
    # Open the ROOT file in read mode
    root_file = TFile(file_path, "read")   #change to UPDATE when you have to write something
    
    # Check if the file is successfully opened
    if not root_file.IsOpen():
        print(f"Error: Unable to open file {file_path}")
        return 0
    print(f"File {file_path} opened successfully!")
    return root_file

def symmetrizeTH2D(histogram):
    hist = histogram.Clone()
    # TODO: First check if the matrix is already symmetric, if yes, don't do anything.
    nBinsX = hist.GetNbinsX()
    nBinsY = hist.GetNbinsY()

    # Ensure the histogram is square
    if nBinsX != nBinsY:
        print("Warning: Histogram is not square! Symmetrization may not make sense.")
        return

    # Loop over the upper triangle of the histogram
    for i in range(1, nBinsX + 1):
        for j in range(i + 1, nBinsY + 1):
            # Get bin contents and errors for (i, j) and (j, i)
            content_ij = hist.GetBinContent(i, j)
            content_ji = hist.GetBinContent(j, i)

            # Average the contents and errors
            symContent = (content_ij + content_ji)

            # Set the symmetrized values back into both (i, j) and (j, i)
            hist.SetBinContent(i, j, symContent)
            hist.SetBinContent(j, i, symContent)


    hist.SetName(hist.GetName()+'_sym')
    hist.SetTitle(hist.GetTitle()+'_sym')
    
    return hist


root_file = open_output("Beam_run.root")
addback_gg_sym = root_file.Get("Addback_gg_sym").Get("All Clovers_sym")
