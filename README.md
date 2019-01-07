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

* rand_mfld_proj/  
  Folder containing code.  
  * run.py  
    Simple functions for generating or plotting data.
  * Other subfolders contain the actual code.

## Building the CPython modules

  You will need to have the appropriate C compilers. On Linux, you should already have them.
  On Windows, [see here](https://wiki.python.org/moin/WindowsCompilers).

  You will need a BLAS/Lapack distribution. Anaconda on Windows sually uses MKL,
  but they recently moved the headers to a different package. You can install it with:
  ```
  > conda install mkl-devel
  ```
  Alternatively, you can downgrade to a version that has the headers, e.g.
  ```
  > conda install mkl=2018.0.3
  ```
  Another option is [OpenBLAS](https://www.openblas.net/)
  ```
  > conda install openblas -c conda-forge
  ```
  ([see here](https://docs.continuum.io/mkl-optimizations/#uninstalling-mkl) under
  Uninstalling MKL).

  If your BLAS/Lapack distribution is installed somewhere numpy isn't expecting,
  you can provide directions in a [site.cfg file](https://github.com/numpy/numpy/blob/master/site.cfg.example).

  Once you have all of the above, you can build the CPython modules in-place:
  ```
  > python setup.py build_ext
  ```
  or you can install it system-wide:
  ```
  > python setup.py install
  ```

## Dependencies

* python3 (only tested with 3.6.3)
* numpy
* matplotlib
* scipy.spatial.distance
* itertools
* math
* collections.abc
* numbers
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
