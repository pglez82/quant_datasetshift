 ## Binary Quantification and Dataset Shift: An Experimental Investigation

 This is the code with the experiments for the paper "Binary Quantification and Dataset Shift: An Experimental Investigation". 

 ### Prerequisites
The recommendation is to install the required dependencies in a separate virtual environment. The main dependency is the quantification library quapy. This is a sequence of steps. Obvioulsy there many more ways of executing the scripts:

```bash
# clone the project
git clone git@github.com:pglez82/quant_datasetshift.git
cd quant_datasetshift
# create the virtual environment
python3 -m venv venv
# activate it
source venv/bin/activate
# update pip
pip install --upgrade pip
# install quapy
pip install git+https://github.com/HLT-ISTI/QuaPy.git@protocols
```
Note that for executing the notebooks it is needed to install `jupyter` and also `seaborn` for generating the figures:

```bash
pip install jupyter seaborn latex
```

 ### Datasets

 Indicate from were the datasets can be downloaded.
 
 ### Description of the scripts
 
 #### Figures and drawings
 The drawing for explanining each of the types of dataset shift where generated with the Jupyter Notebook `datasetshift_drawings.ipynb`.

 #### Preliminary analysis of data
 A preliminary analysis of data was carried out in the Jupyter Notebook `premiliminary_analysis_reviews.ipynb`. This notebook also generates the table with the description of the dataset.

 #### Main experiments
 Experiments were carried out with python script and the subsequent analysis of the data generated, was carried out using Jupyter Notebooks. Note that the each notebook will automatically generate all the tables in latex code and the figures that are finally taken into the paper.

 - **Prior probability shift**: Script `reviews_prior_shift.py` and notebook `analyze_results_prior.ipynb`.
 - **Change in prior**: Script `reviews_changeprior.py` and notebook `analyze_results_changeprior.ipynb`.
 - **Covariate shift**: Script `reviews_covariate_shift.py` and notebook `analyze_results_covariate.ipynb`.
 - **Concept shift**: Script `reviews_concept_shift.py` and notebook `analyze_results_concept.ipynb`.
 - **Concept shift v2**: Script `reviews_concept_shift_v2.py` and notebook `analyze_results_concept_v2.ipynb`.
 TODO: remove one of the two concept shift experiments depending on what we put in the paper.

 Note that these python scripts may require some directories to be created, in particular, the directory `results` with a subdirectory for each experiment `prior`, `changeprior`, `covariate` and `concept`. The notebooks also need some directories created, in particular, `tables` and `images`, where latex tables and figures will be generated. Datasets must be placed under the `dataset/reviews` directory.

