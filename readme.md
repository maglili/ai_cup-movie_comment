# AI-CUP

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Generic badge](https://img.shields.io/badge/Model-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Plotting-passing-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/dataset-passing-green.svg)](https://shields.io/)

[<img src="https://ForTheBadge.com/images/badges/made-with-python.svg" alt="made with python" width="" height="30px">](https://www.python.org/)
[<img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="made with jupyter" width="px" height="30px">](https://jupyter.org/try)
[<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" width="px" height="30px">](https://pytorch.org/)
[<img src="https://ForTheBadge.com/images/badges/uses-git.svg" alt="use-git" width="px" height="30px">](https://git-scm.com/)

## Intro

**Train:**

```bash
python main.py -m train -bs 8 -epo 10 --model_name bert-base-cased
```

**Test:**

```bash
python main.py -m test -bs 8 -epo 10 --model_name bert-base-cased
```

**Predict:**

```bash
python main.py -m predict -bs 8 -epo 10 --model_name bert-base-cased
```

## Reference

1. [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#a2-weight-decay)
2. [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
3. [Cross validation strategy when blending/stacking](https://www.kaggle.com/general/18793)
4. [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
5. [4 Boosting Algorithms You Should Know – GBM, XGBoost, LightGBM & CatBoost](https://www.analyticsvidhya.com/blog/2020/02-boosting-algorithms-machine-learning/)
