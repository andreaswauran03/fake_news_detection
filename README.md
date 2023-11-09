---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.4
  nbformat: 4
  nbformat_minor: 5
---

## Fake News Detection Project
```
We are proud to present our fake news detection project! In this project, we conducted an analysis of news data in English and grouped it into two categories of interest: fake news and real news. The goal is to help identify and understand the differences between the two, as well as raise awareness of the increasingly troubling problem of fake news in today's information world. Join us on this journey to unearth the truth and analyze news accuracy!
```

## Setup Installation
### Setup Environment

``` python
conda create --name main-ds python=3.11
conda activate main-ds
pip install numpy pandas scikit-learn jupyter joblib streamlit
```

### Run Steamlit App for Webapps

``` python
streamlit run web_apps.py
```