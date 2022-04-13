# digital-core 

**digital-core toolkit** - ML workflows for core data. Project in active development.   

&nbsp;

## Getting started 

Install 
```shell
conda env create -f environment.yml
```


See `demos` for example workflows.    

+ Geochemistry clustering workflow - `dGeochemistry.ipynb`  
+ Hyperspectral mineralmap extraction workflow - `dHyperspectral.ipynb`

 &nbsp;

## Workflow sessions

Import a geochemistry workflow session 
```python
from digitalcore import GeochemML
```

Import mineralmap workflow session
```python
from digitalcore import MineralMap
```

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
