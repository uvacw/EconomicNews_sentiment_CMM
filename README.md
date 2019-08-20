# About the repo

This is a collection of scripts used to generate the results for 
_"What’s the Tone? Easy Doesn’t Do It: Analyzing Performance and Agreement Between Off-the-Shelf Sentiment Analysis Tools"_.

**FAIR WARNING**: This code is meant to show the analysis process, and is not in any way shape or form 
suitable for production (_hint_: there is no proper package layout). 

To provide an relatively easy overview of revision-to-revision changes, files and folders have been marked "r1" and "r2"
for post-first and post-second review changes respectivily. 

Some dependencies are non-Python:
- The popular `sentistrength` distribution used the .Jar freely available for academic use [here](http://sentistrength.wlv.ac.uk/
)
- The readability score package used here can be installed from github using 
```pip install git+https://github.com/wimmuskee/readability-score.git```
- R is used to analyze the inter-rater scores, code can be found in the .Rmd files

### Analysis as published:
- main analyses in `CMM_R2-The good and bad of Economy analysis.ipynb`
  - helper functions in `scripts_r2/`
  - scatterplots of results in `scatterplots_r2/`
- inter-rater scores in calculated in `CMM_R1_Kalpha_ordinal.Rmd`
  - plots generated using `CMM_R2_scatterplots.Rmd`
  - csv files with inter-rater results in `kalpha_results_r2`
- `data/` contains the raw data  
- `results/data_with_sentiment.csv` contains the calculated sentiment scores used in the analyses

# Installation:

## 1. set up a virtualenv

You are advised to use a virtual environment to keep dependencies of this project separate from your global
environment. You can use your favorite solution, but below we demonstrate installation using the anaconda Python
distribution. 

```bash
# Create the environment (do this only once)
conda create -n economic_sentiment python=3
# Activate the environment
source activate economic_sentiment
```
## 2. Clone the repo

Get the codebase from Github.

```bash
git clone https://github.com/bobvdvelde/economic_sentiment.git
cd economic_sentiment
```

## 3. Install dependencies

You can install most of the Python dependencies from Pypy by using `pip`. 

```bash
pip install -r Requirements
pip install git+https://github.com/wimmuskee/readability-score.git
python -c "import nltk; nltk.download('punkt')"
```

Note that dependency drift is an issue with some of these libraries.

# How to run:

You can find the latest results in `CMM_R2-The good and bad of Economy analysis.ipynb` which presents all the results.
The underlying analysis scripts are in `scripts_r2` (note that the "r2" denotes this file was changed in response to the
second round of reviews). 

To reproduce the sentiment scores used in the paper:

```{python}
from scripts_r1.analyze import load_files, add_sentiments
d = load_files("data")
add_sentiments(d, "title")
d.to_csv("out.csv")
```
