# CASTRO-5G

This Open Source repo presents a collection of 5G sparse multipath channel simulation, estimation and location signal processing tools in Python. CASTRO-5G is a repository started as a means to publish the software results of the research project Communications And Spatial Tracking RatiO (CASTRO) funded by Spanish Ministerio de Ciencia e Innovación (MICINN) - Proyectos de Generación de Conocimiento 2021 PID2021-122483OA-I00.

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

## tests

TBW

## Structure of the Code and Tutorials

TBW

## Deployment

TBW

## Built With

### Code

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

### Academic Results

TBW

## Contributing

Please read [CONTRIBUTING.md](https://github.com/gomezcuba/CASTRO-5G/blob/main/CONTRIBUTE.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versions

* Version 0.1 - initialized a private github repo with [gomezcuba](https://github.com/gomezcuba)'s draft code, which was still very deficient
* Version 0.2 [this release] - [gomezcuba](https://github.com/gomezcuba) prepared the repo for going public
* Version 0.3 [upcoming]

## Authors

* **Felipe Gomez Cuba** - *Initial work* - [gomezcuba](https://github.com/gomezcuba) [website](https://www.felipegomezcuba.info/)
* **Gonzalo Feijoo Rodriguez (TFG)** - *Multipath Location module v2 and tutorials* - [gonzaFero](https://github.com/gonzaFero)
* **Iago Dafonte (TFG)** - *3GPP Channel Generator module v2 and example files* - [iagoalvarez](https://github.com/iagoalvarez)

See also the list of [contributors](https://github.com/your/project/contributors) actively submitting to this project.

## License

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details

## Acknowledgments

TBW
