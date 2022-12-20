# CASTRO-5G

This Open Source repo presents a collection of 5G sparse multipath channel simulation, estimation and location signal processing tools in Python. CASTRO-5G is a repository started as a means to publish the software results of the research project Communications And Spatial Tracking RatiO (CASTRO) funded by Spanish Ministerio de Ciencia e Innovación (MICINN) - Proyectos de Generación de Conocimiento 2021 PID2021-122483OA-I00.

## How to contribute

Please read [CONTRIBUTING.md](https://github.com/gomezcuba/CASTRO-5G/blob/main/CONTRIBUTE.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versions

* Version 0.0.1 - initialized a private github repo with [gomezcuba](https://github.com/gomezcuba)'s draft code, which was still very deficient
* Version 0.0.2 - [gonzaFero](https://github.com/gonzaFero) finished his TFG, upgrading MultipathLocationEstimator.py and creating its tutorials
* Version 0.1.0 (this release) - [gomezcuba](https://github.com/gomezcuba) prepared the repo for going public, 
* Version 0.1.1 (upcoming) - [iagoalvarez](https://github.com/iagoalvarez) is finishing his TFG, upgrading threeGPPMultipathGenerator.py and creating its test files

## Authors

* **Felipe Gomez Cuba** - *Initial work* - [gomezcuba](https://github.com/gomezcuba) [website](https://www.felipegomezcuba.info/)
* **Gonzalo Feijoo Rodriguez (TFG)** - *Multipath Location module v2 and tutorials* - [gonzaFero](https://github.com/gonzaFero)
* **Iago Dafonte (TFG)** - *3GPP Channel Generator module v2 and example files* - [iagoalvarez](https://github.com/iagoalvarez)

See also the list of [contributors](https://github.com/your/project/contributors) actively submitting to this project.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Our project aims to maintain dependencies low. On a standard python3 install, you would only need the following dependencies:

* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* (Our roadmap foresees including [Pandas](https://pandas.pydata.org/) as a default dependency, as it can enable a very elegant storage of multipath data. This may become a prerequisite in the future. Currently, the code can be run without this dependency. However, we highly recommend to have Pandas installed from the begining of your work, specially if you plan to upgrade to next versions of CASTRO-5G)

### Installing

You may simply download the code into a folder or clone the repo

```
git clone https://github.com/gomezcuba/CASTRO-5G.git
```

You can test that dependencies are met by running the *raygeometry.py* simulation

```
python raygeometry.py
```

the expected result is that this script should run without warnings, print several debug lines, and generate several .eps results files in the working folder.

## Structure of the Code

TBW

## Tests, Examples and Tutorials

TBW

## Deployment

The following files can be added to your PATH and employed as python libraries or shell commands

* multipathChannel.py
* threeGPPMultipathGenerator.py
* OMPCachedRunner.py
* mpraylocsim.py

##  Acknowledgments

### OSS

* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Spyder3](https://www.spyder-ide.org/)

### Featured academic Results

* F. Gómez-Cuba and A. J. Goldsmith, "Compressed Sensing Channel Estimation for OFDM With Non-Gaussian Multipath Gains," in IEEE Transactions on Wireless Communications, vol. 19, no. 1, pp. 47-61, Jan. 2020
* F. Gómez-Cuba, "Compressed Sensing Channel Estimation for OTFS Modulation in Non-Integer Delay-Doppler Domain," 2021 IEEE Global Communications Conference (GLOBECOM), 2021,
* 3GPP. (2022). 3rd Generation Partnership Project; Technical Specification Group Radio Access Network; Study on channel model for frequencies from 0.5 to 100 GHz (Release 17). ETSI TR 38.901, 17.0.

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details

## Acknowledgments

TBW
