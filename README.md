# WiPhy
<img align="right" width="150px" height="150px" src="https://user-images.githubusercontent.com/62990567/87901098-16cfb080-ca91-11ea-99fb-8f46db3c741d.png">
WiPhy is an open-source Python package for wireless signal processing at the physical layer, and is mainly developed by Ishikawa Laboratory, Yokohama National University, Japan. This package attempts to facilitate reproducible research in wireless communications. 
It supports classical wireless technologies, such as MIMO and OFDM, and state-of-the-art technologies, such as index modulation and nonsquare differential coding. This project is derived from <a href="https://github.com/ishikawalab/imtoolkit" target="_blank">IMToolkit</a>, which specializes in index modulation.

## Key Features
The major advantages of this package are highlighted as follows:
- **Incredibly fast.** It accelerates bit error ratio and average mutual information simulations by using a state-of-the-art Nvidia GPU and the massively parallel algorithms proposed in [1].
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

The above installation process requires NumPy, Pandas, SciPy, SymPy, Numba, and tqdm, all of which are popular Python packages.
Additionally, it is strongly recommended to install [CuPy](https://cupy.chainer.org/) 5.40+. 
WiPhy is heavily dependent on CuPy to achieve significantly fast Monte Carlo simulations.
[The key components required by CuPy are listed here.](https://docs-cupy.chainer.org/en/stable/install.html)
In case CuPy is not installed in your environment, WiPhy uses NumPy only.

## Citations

It would be highly appreciated if you cite the following reference when using WiPhy.

- [1] N. Ishikawa, ``[IMToolkit: An open-source index modulation toolkit for reproducible research based on massively parallel algorithms](https://doi.org/10.1109%2Faccess.2019.2928033),'' IEEE Access, vol. 7, pp. 93830--93846, July 2019.

Of course, if your project relies on CuPy, the following reference is strongly recommended.

- [2] R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Loomis, ``[CuPy: A NumPy-compatible library for NVIDIA GPU calculations](http://learningsys.org/nips17/assets/papers/paper_16.pdf),'' in Conference on Neural Information Processing Systems Workshop, Long Beach, CA, USA, December 4--9, 2017.
