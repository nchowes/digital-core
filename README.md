# digital-core 

**digital-core toolkit** - ML workflows for core data. Project in active development.     
[![CircleCI](https://circleci.com/gh/nchowes/digital-core/tree/dev_geochem_simple.svg?style=svg&circle-token=4269d33c29007ef65014893358fc16b2a8555e82)](https://circleci.com/gh/nchowes/digital-core/?branch=dev_geochem_simple)
&nbsp;

## Getting started 

Install 
```shell
conda env create -f environment.yml
```

 &nbsp;

 ## Demos 

See `demos` for example workflows.    

+ Geochemistry clustering workflow - `dGeochemistry.ipynb`  
+ Hyperspectral mineralmap extraction workflow - `dHyperspectral.ipynb`


 &nbsp;


## Test

To run package tests from base directory
```shell
pytest -v -c./tests/pytest.ini
```

To dry-run and notebook test configuration (not run)
```shell 
pytest -v -c ./tests/pytest.ini --collect-only --nbmake ./demos \
--ignore=demos/dAutoML.ipynb \
--ignore=demos/dUnsupervised.ipynb
```

To run notebook tests (smoke tests) from base directory, ignoring certain notebooks. 
```shell
pytest -v -c ./tests/pytest.ini --nbmake ./demos \
--ignore=demos/dAutoML.ipynb \
--ignore=demos/dUnsupervised.ipynb
```
