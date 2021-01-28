# WiPhy
<img align="right" width="150px" height="150px" src="https://user-images.githubusercontent.com/62990567/87901098-16cfb080-ca91-11ea-99fb-8f46db3c741d.png">
WiPhy is an open-source Python package for wireless signal processing at the physical layer, and is mainly developed by Ishikawa Laboratory, Yokohama National University, Japan. This package attempts to facilitate reproducible research in wireless communications. 
It supports classical wireless technologies, such as MIMO and OFDM, and state-of-the-art technologies, such as index modulation and nonsquare differential coding. This project is derived from <a href="https://github.com/ishikawalab/imtoolkit" target="_blank">IMToolkit</a>, which specializes in index modulation.

## Key Features
The major advantages of this package are highlighted as follows:
- **JIT friendly.** It dose not rely on any class and object-oriented programming, which is different from the conventional IMToolkit. It is suitable for both PyPy and Numba. Users are free from the nightmare of complex @jitclass decorations.
- **Highly reliable.** It has been maintained based on test-driven development, as with other high-quality packages. The simulation results have been used by IEEE journal papers.

## Disadvantages
- Some methods are not well documented.  

## Installation Guide
WiPhy is available from the Python official package repository [PyPi](https://pypi.org/project/imtoolkit/).

    > pip install wiphy

The WiPhy development team welcomes other researchers' contributions and pull requests.
In that case, it would be better to install the latest package and activate the editable mode as follows:

    > git clone https://github.com/ishikawalab/wiphy/
    > pip install -e ./wiphy # this activates the editable mode

If you use Anaconda, you can install WiPhy as follows.

    > git clone https://github.com/ishikawalab/wiphy/
    > conda develop ./wiphy

The Anaconda version 4.6.0 or above may also allow the following installation commands.

    > conda config --set pip_interop_enabled True
    > # for typical users
    > pip install wiphy
    > # for developers
    > git clone https://github.com/ishikawalab/wiphy
    > pip install -e ./wiphy

The above installation process requires NumPy, Pandas, SciPy, SymPy, Numba, and tqdm, all of which are popular Python packages.

## Citations

It would be highly appreciated if you cite the following reference when using WiPhy.

- [1] N. Ishikawa, ``[IMToolkit: An open-source index modulation toolkit for reproducible research based on massively parallel algorithms](https://doi.org/10.1109%2Faccess.2019.2928033),'' IEEE Access, vol. 7, pp. 93830--93846, July 2019.
