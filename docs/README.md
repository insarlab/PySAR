[![Language](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Docs Status](https://readthedocs.org/projects/mintpy/badge/?version=latest&style=flat-square)](https://mintpy.readthedocs.io/?badge=latest)
[![CircleCI](https://img.shields.io/circleci/build/github/insarlab/MintPy.svg?logo=circleci&label=tests&style=flat-square)](https://circleci.com/gh/insarlab/MintPy)
[![Docker Status](https://img.shields.io/github/actions/workflow/status/insarlab/MintPy/build-docker.yml?label=docker&style=flat-square&logo=docker&logoColor=white)](https://github.com/insarlab/MintPy/pkgs/container/mintpy)
[![Conda Download](https://img.shields.io/conda/dn/conda-forge/mintpy?color=green&style=flat-square&label=conda%20downloads)](https://anaconda.org/conda-forge/mintpy)
[![Version](https://img.shields.io/github/v/release/insarlab/MintPy?color=yellow&label=version&style=flat-square)](https://github.com/insarlab/MintPy/releases)
[![Forum](https://img.shields.io/badge/forum-Google%20Groups-orange.svg?style=flat-square)](https://groups.google.com/g/mintpy)
[![License](https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square)](https://github.com/insarlab/MintPy/blob/main/LICENSE)
[![Citation](https://img.shields.io/badge/doi-10.1016%2Fj.cageo.2019.104331-blue?style=flat-square)](https://doi.org/10.1016/j.cageo.2019.104331)

## MintPy ##

The Miami INsar Time-series software in PYthon (MintPy as /mɪnt paɪ/) is an open-source package for Interferometric Synthetic Aperture Radar (InSAR) time series analysis. It reads the stack of interferograms (coregistered and unwrapped) in [ISCE](https://github.com/isce-framework/isce2), [ARIA](https://github.com/aria-tools/ARIA-tools), [FRInGE](https://github.com/isce-framework/fringe), [HyP3](https://hyp3-docs.asf.alaska.edu/), [GMTSAR](https://github.com/gmtsar/gmtsar), [SNAP](http://step.esa.int/), [GAMMA](https://www.gamma-rs.ch/software) or ROI_PAC format, and produces three dimensional (2D in space and 1D in time) ground surface displacement in line-of-sight direction. It includes a routine time series analysis (`smallbaselineApp.py`) and some independent toolbox.

This package was called PySAR before version 1.1.1. For version 1.1.2 and onward, we use MintPy instead.

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

### 1. [Installation](https://mintpy.readthedocs.io/en/latest/installation.html) ###

### 2. Running MintPy ###

#### 2.1 Routine workflow `smallbaselineApp.py` ####

MintPy reads a stack of interferograms (unwrapped interferograms, coherence and connected components from SNAPHU if available) and the geometry files (DEM, lookup table, incidence angle, etc.). You need to give the [path to where the files are](https://mintpy.readthedocs.io/en/latest/dir_structure.html) and MintPy takes care of the rest!

```bash
smallbaselineApp.py                         # run with default template 'smallbaselineApp.cfg'
smallbaselineApp.py <custom_template>       # run with default and custom templates
smallbaselineApp.py -h / --help             # help
smallbaselineApp.py -H                      # print    default template options
smallbaselineApp.py -g                      # generate default template if it does not exist
smallbaselineApp.py -g <custom_template>    # generate/update default template based on custom template

# Run with --start/stop/dostep options
smallbaselineApp.py GalapagosSenDT128.txt --dostep velocity  # run step 'velocity' only
smallbaselineApp.py GalapagosSenDT128.txt --end load_data    # end run after step 'load_data'
```

Inside smallbaselineApp.py, it reads the unwrapped interferograms, references all of them to the same coherent pixel (reference point), calculates the phase closure and estimates the unwrapping errors (if it has been asked for), inverts the network of interferograms into time-series, calculates the temporal coherence to evaluate the quality of inversion, corrects local oscillator drift (for Envisat only), corrects stratified tropospheric delay (using global atmospheric models or phase-elevation-ratio approach), removes phase ramps (if it has been asked for), corrects DEM error,... and finally estimates the velocity.

Configuration parameters for each step are initiated with default values in a customizable text file [**smallbaselineApp.cfg**](https://mintpy.readthedocs.io/en/latest/defaults/smallbaselineApp.cfg).

#### [Example](https://mintpy.readthedocs.io/en/latest/demo_dataset.html) on Fernandina volcano, Galápagos with Sentinel-1 data ####

```bash
wget https://zenodo.org/record/3952953/files/FernandinaSenDT128.tar.xz
tar -xvJf FernandinaSenDT128.tar.xz
cd FernandinaSenDT128/mintpy
smallbaselineApp.py ${MINTPY_HOME}/docs/templates/FernandinaSenDT128.txt
```

<p align="left">
  <img width="600" src="https://insarlab.github.io/figs/docs/mintpy/FernandinaSenDT128-ISCE2.jpg">
</p>

Results are plotted in **./pic** folder. To explore more data information and visualization, try the following scripts:

```bash
info.py                    # check HDF5 file structure and metadata
view.py                    # 2D map view
tsview.py                  # 1D point time-series (interactive)
plot_coherence_matrix.py   # plot coherence matrix for one pixel (interactive)
plot_network.py            # plot network configuration of the dataset
plot_transection.py        # plot 1D profile along a line of a 2D matrix (interactive)
save_kmz.py                # generate Google Earth KMZ file in points or raster image
save_kmz_timeseries.py     # generate Google Earth KMZ file in points for time-series (interactive)
```

#### 2.2 Customized processing recipe ####

MintPy is a toolbox with individual utility scripts. Simply run the script with `-h / --help` to see its usage, you could build your own customized processing recipe! [Here](https://mintpy.readthedocs.io/en/latest/scripts/compare_velocity_with_diff_tropo.sh) is an example to compare the velocities estimated from displacement time-series with different tropospheric delay corrections.

#### 2.3 Build on top of `mintpy` module ####

MintPy is modulized in Python with utilities classes and functions and well commented in the code level. Users who are familiar with Python could build their own functions and modules on top of [`mintpy.objects`](https://mintpy.readthedocs.io/en/latest/objects.html) and [`mintpy.utils`](https://mintpy.readthedocs.io/en/latest/utils.html). However, we don't have a complete API document website yet (maybe you can contribute this!). Below is an example of reading the 3D matrix of displacement time-series from an HDF5 file.

```python
from mintpy.utils import readfile
ts_data, meta = readfile.read('timeseries_ERA5_ramp_demErr.h5')
```

### 3. [Documentation](https://mintpy.readthedocs.io/) ###

Algorithms implemented in the software are described in details at [Yunjun et al. (2019)](https://doi.org/10.1016/j.cageo.2019.104331).

+ [Quick start with example datasets](https://mintpy.readthedocs.io/en/latest/demo_dataset.html)
+ [Example data directory](https://mintpy.readthedocs.io/en/latest/dir_structure.html)
+ [Example template files](https://mintpy.readthedocs.io/en/latest/templates/README.html)
+ [Tutorials in Jupyter Notebook](https://github.com/insarlab/MintPy-tutorial)

### 4. Contact us ###

+ Most development discussion happens on GitHub. Feel free to [open an issue](https://github.com/insarlab/MintPy/issues) or comment on any open issue or pull request.
+ Join our [user forum on google groups](https://groups.google.com/g/mintpy) or use [github discussions](https://github.com/insarlab/MintPy/discussions) to ask questions or leave comments.

### 5. Contributing ###

**Imposter syndrome disclaimer:** We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not ready to be an open source contributor; that your skills aren't nearly good enough to contribute. What could you possibly offer?

We assure you - the little voice in your head is wrong. If you can write code at all, you can contribute code to open source. Contributing to open source projects is a fantastic way to advance one's coding skills. Writing perfect code isn't the measure of a good developer (that would disqualify all of us!); it's trying to create something, making mistakes, and learning from those mistakes. That's how we all improve, and we are happy to help others learn.

**Being an open source contributor doesn't just mean writing code.** You can help out by writing or proofreading documentation, suggesting or implementing tests, or even giving feedback about the project (and yes - that includes giving feedback about the contribution process). Some of these contributions may be the most valuable to the project as a whole, because you're coming to the project with fresh eyes, so you can see the errors and assumptions that seasoned contributors have glossed over.

For more information, please read our [contributing guide](https://mintpy.readthedocs.io/en/latest/CONTRIBUTING.html).

_This disclaimer was adapted from the [MetPy project](https://github.com/Unidata/MetPy)._

### 6. Citing this work ###

Yunjun, Z., Fattahi, H., and Amelung, F. (2019), Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction, _Computers & Geosciences_, _133_, 104331. [ [doi](https://doi.org/10.1016/j.cageo.2019.104331) \| [arxiv](https://doi.org/10.31223/osf.io/9sz6m) \| [data](https://doi.org/10.5281/zenodo.3464190) \| [notebook](https://github.com/geodesymiami/Yunjun_et_al-2019-MintPy) ]

In addition to the above, we recommend that you cite the original publications that describe the algorithms used in your specific analysis. They are noted briefly in the [default template file](https://mintpy.readthedocs.io/en/latest/defaults/smallbaselineApp.cfg) and listed in the [reference file](https://mintpy.readthedocs.io/en/latest/references.html).