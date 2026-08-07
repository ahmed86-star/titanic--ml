# Titanic Survival Prediction: An Honest ML Workflow

Final project for my machine learning course at Franklin University. Binary
classification on the Kaggle Titanic dataset (891 labeled passengers), built
around one question:

> How much of the predictive performance comes from the data itself, versus
> the modeling machinery stacked on top of it?

## Background

Before this project I wrote a peer review of the most popular community
solution to this problem, Yassine Ghouzam's "Titanic Top 4% with Ensemble
Modeling." The workflow in that notebook is complete and well explained, but
it has six methodological problems that thousands of beginners have absorbed
from it:

1. Training and test data are preprocessed together, which leaks test-set
   statistics (medians, encoding vocabularies) into the model.
2. Ten real passengers are deleted as "outliers" with no experiment to show
   deletion helps.
3. Scale-sensitive models (SVC, KNN, logistic regression) are benchmarked
   against tree models on unscaled data.
4. Roughly sixty sparse ticket and cabin dummy columns are added to a
   dataset with fewer than 900 rows.
5. Accuracy is the only metric reported. No confusion matrix, no ROC curve,
   no error analysis.
6. Cross-validation scores are reported from the same folds used to tune
   hyperparameters, which makes them optimistic by construction.

This repository implements the corrected version of each of those points and
measures what the corrections reveal.

## Method

The full workflow is in `notebooks/Ahmed_ML_Final_Project.ipynb`, executed
end to end with outputs embedded. The short version:

- **Leakage-free preprocessing.** Every fitted transform (imputation,
  log-fare, scaling, one-hot encoding) lives inside a scikit-learn
  `Pipeline` behind a `ColumnTransformer`, so cross-validation refits it on
  training folds only. The grouped-median age imputation from the original
  notebook is kept, but reimplemented as a proper fitted transformer.
- **Feature engineering.** Title extracted from the name string, family size
  and binned family categories, and a single `CabinKnown` flag replacing the
  original notebook's deck and ticket dummies. EDA in the notebook shows the
  missingness of `Cabin` is itself predictive, so one binary flag captures
  the signal without the dimensionality cost.
- **Evaluation protocol.** Stratified 80/20 split. All model comparison and
  grid search happens on the development set with stratified 10-fold CV. The
  held-out test set is touched exactly once, at the end, to produce the
  reported estimate.
- **Baseline ladder.** Majority-class dummy, logistic regression,
  depth-limited random forest, histogram gradient boosting. Each step up in
  complexity has to justify itself against the previous one.
- **Outlier ablation.** The Tukey IQR deletion from the original notebook is
  tested instead of assumed: same pipeline, same folds, with and without the
  flagged passengers.

## Results

| Model | Score | Evaluation |
|---|---|---|
| Majority-class baseline | 61.7% | Dev 10-fold CV |
| Logistic regression (untuned) | 82.6% | Dev 10-fold CV |
| Hist gradient boosting (tuned) | 84.3% | Dev 10-fold CV |
| Hist gradient boosting (tuned) | **79.3%** | **Held-out test, single shot** |
| Ghouzam five-model ensemble | ~80-82% | Public leaderboard (reference) |

Held-out ROC AUC: 0.84. Survivor-class precision 0.76, recall 0.68.

Three findings worth pulling out:

**The CV optimism gap showed up on schedule.** Tuned development CV said
84.3%; the untouched test set said 79.3%. That is the same three-to-four
point gap visible in the original notebook (0.83 CV against a 0.80
leaderboard score), except here the protocol was designed to expose it
rather than hide it. The 79.3% is a less flattering number than the CV
score, but it is the honest one.

**Deleting outliers was never justified.** Keeping the Tukey-flagged
passengers scored 84.3% CV; deleting them scored 84.0%. The difference is
well inside one standard deviation of the CV estimate. The flagged rows are
real members of the population, mostly large families with unusual fares, so
the defensible default is to keep them.

**The remaining errors are concentrated and mostly irreducible.** Error
analysis on the test set shows misclassifications cluster in the
against-the-odds groups: third-class women who died (32% error rate) and
first-class men who survived (43% error rate). These passengers break the
dominant sex-by-class pattern, and no amount of ensemble machinery fixes
that, because the deciding factor was largely luck during the sinking.

The answer to the research question: almost all of the performance comes
from the data. A plain logistic regression on properly prepared features
lands within about two points of tuned gradient boosting and within about
one point of the original notebook's five-model ensemble. The strong signals
are sex, class, title, family structure, and the informativeness of a
missing cabin record. The machinery on top buys marginal gains at a real
cost in interpretability and tuning time.

## Repository layout

```
notebooks/
  Ahmed_ML_Final_Project.ipynb   Full workflow, executed with outputs
data/
  titanic.csv                    Kaggle training set (891 rows, train.csv schema)
figures/
  fig_01.png ... fig_04.png      Figures exported from the notebook
requirements.txt
```

## Running it

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Ahmed_ML_Final_Project.ipynb
```

The notebook expects `titanic.csv` in its working directory; either run it
from `data/` on your path or adjust the single `read_csv` line. Kaggle's
`train.csv` is a drop-in replacement. Full run takes under two minutes on a
laptop; the random seed is fixed, so results reproduce exactly.

## Limitations

- 891 rows means CV estimates carry real variance (standard deviations of
  two to four points). Small differences between models are not significant,
  and the notebook says so rather than ranking them anyway.
- The held-out test set is 179 rows, so the final estimate has a wide
  confidence band. It is unbiased, which the tuned CV scores are not, but it
  is not precise.
- Findings are specific to this dataset. The workflow is what generalizes.

## References

- Ghouzam, Y. Titanic Top 4% with Ensemble Modeling. Kaggle.
- Kaggle. Titanic: Machine Learning from Disaster (competition dataset).
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  JMLR, 12, 2825-2830.
- Ahmed, A. (2026). Case Study Research Report: A Peer Review of "Titanic
  Top 4% with Ensemble Modeling." Franklin University.
