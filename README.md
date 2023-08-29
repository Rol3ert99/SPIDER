# SPIDER
SPIDER is a preprocessing algorithm for imbalanced datasets designed to improve the quality of datasets
by improving their representation of minority classes and removing noisy examples. The algorithm uses 
various amplification strategies based on nearest neighbor relationships to achieve this goal.
The algorithm was implemented on the basis of publication [1].


## Installation
To use SPIDER, you can clone the repository:

```bash
git clone https://github.com/Rol3ert99/Spider.git
cd Spider
```
Once you have cloned the repository and navigated to the "Spider" directory, you can install SPIDER using pip:
```bash
pip install .
```
This command will install the Spider package, allowing you to import and use the SPIDER class in your Python scripts.


## Usage
Here are three examples of using Spider:

1. Weak Amplification:
```python
from Spider import SPIDER

# Initialize Spider with the desired amplification type
spider = SPIDER(amplification_type='weak_amplification')

# Load your dataset and labels
X = ...  # Your feature matrix
y = ...  # Your target labels

# Apply the preprocessing
new_X, new_y = spider.fit_resample(X, y)
```

2. Weak Amplification with Relabeling:
```python
from Spider import SPIDER

# Initialize Spider with weak amplification and relabeling
sp2 = SPIDER(amplification_type='weak_amplification_with_relabeling')

# Load your dataset and labels
X = ...  # Your feature matrix
y = ...  # Your target labels

# Apply the preprocessing
new_X, new_y = sp2.fit_resample(X, y)
```

3. Strong Amplification:
```python
from Spider import SPIDER

# Initialize Spider with strong amplification
sp2 = SPIDER(amplification_type='strong_amplification')

# Load your dataset and labels
X = ...  # Your feature matrix
y = ...  # Your target labels

# Apply the preprocessing
new_X, new_y = sp2.fit_resample(X, y)
```


## Running Tests
Assume that pip Spider package is installed.
To run tests for Spider, navigate to the project directory and execute:
```bash
python test.py
```


## Publication
For more details on the Spider algorithm, you can refer to the following publication:  
- [Selective Pre-processing of Imbalanced Data for Improving Classification Performance](https://link.springer.com/chapter/10.1007/978-3-540-85836-2_27)



















