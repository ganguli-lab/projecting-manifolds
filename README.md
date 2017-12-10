# projecting-manifolds
Code for paper: Lahiri, Gao, Ganguli, "Random projections of random manifolds".

* Run 'one-click.py' to generate and save all data and plots.  

* Run 'demo-data.py' to quickly generate example data for plots.  

* Run 'demo-plots.py' to make plots without saving.  
  (Un)comment lines to choose whether to use quick or full data.

* Data/  
  Folder containing generated data for plots.  
  We have provided example data.

* Figures/  
  Folder containing plots.

* RandProjRandMan/  
  Folder containing code.  
  * run.py  
    Simple functions for generating or plotting data.
  * Other subfolders contain the actual code.

## Dependencies

* python3 (only tested with 3.6.3)
* numpy
* matplotlib
* matplotlib.pyplot
* scipy.spatial.distance
* itertools
* math
* collections.abc
* typing

## Type hints

The following aliases are used:

    Styles = Sequence[Mapping[str, str]]
    StyleSet = Mapping[str, Styles]
    Options = Mapping[str, Any]
    OptionSet = Mapping[str, Options]
    Labels = Sequence[str]
    LabelSet = Mapping[str, Labels]
    Axes = mpl.axes.Axes
    Figure = mpl.figure.Figure
    Lines = Sequence[mpl.lines.Line2D]
