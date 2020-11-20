# msc_thesis
Master Thesis

# Installation
Conda packages
```
matplotlib==3.0.2 # conda
pandas==0.24.2 5 # conda
numpy==1.15.4 # conda
coloredlogs==10.0 # conda

nltk==3.5 # conda installed
unidecode==1.1.1 # conda installed
```
Conda-forge packages
```
torch==1.3.1 # conda, install as pytorch==1.3.1
spacy==2.2.3 # conda-forge <- error (dependencies) spacy-2.0.12 works using conda
```
Pip packages
```
subword_nmt==0.3.7 # pip
torchtext==0.6.0 # pip
rouge==0.3.2 # pip
```

Load spacy model
```
python -m spacy download en_core_web_sm
```
