# EPL Match Outcome Predictor

A stacking ensemble that predicts English Premier League results (Home / Draw / Away) using only information available before kick-off. Built for UCL's COMP0036 group coursework, *Beat the Bookie* (December 2025).

**Headline result:** 56.2% accuracy on a held-out, chronologically later test set (1,530 matches) vs a ~53% bookmaker benchmark.

The main weakness is draws: ensemble draw recall is only 13.7%. See [Results](#results) and [Limitations](#limitations).

---

## Contents

- [Approach](#approach)
- [What's different about this approach](#whats-different-about-this-approach)
- [Results](#results)
- [Key findings](#key-findings)
- [Limitations](#limitations)
- [Contributions](#contributions)
- [Repository layout](#repository-layout)
- [Reproducing](#reproducing)
- [Data sources](#data-sources)

---

## Approach

![Model architecture: base learners (auxiliary closeness model, Poisson, CatBoost, XGBoost) feed three meta-learners (Home, Away, Draw), whose outputs pass through a softmax to give the final prediction](/figures/architecture.png)

*Raw match data is turned into features. An auxiliary closeness model feeds CatBoost and XGBoost. Poisson, CatBoost and XGBoost output class probabilities, and three meta-learners (one per outcome) combine them. A softmax over the meta-learner outputs gives the final H / D / A prediction.*

- **Poisson model:** time-weighted attack and defence strengths with a home-advantage factor, giving interpretable baseline H/D/A probabilities.
- **CatBoost:** native categorical handling (teams, referees) with balanced class weights.
- **XGBoost:** label-encoded features, L1/L2 regularisation, and a 1.3× draw-class weight to improve minority-class recall.
- **P(Close) auxiliary model:** predicts whether a match is decided by ≤ 1 goal, trained out-of-fold and fed to the classifiers as a feature to help flag draw-prone fixtures.
- **Three-part meta-learner:** separate binary logistic regressions (Platt-calibrated) per outcome, plus a tunable additive draw boost applied before `argmax`.

### Evaluation design

- **Chronological 70 / 15 / 15 split** (train / validation / test), sorted by date. The test set is the most recent 15% of historical matches and was used only for final evaluation.
- **Out-of-fold stacking:** the meta-learner is trained only on predictions made by base models that had not seen those rows.
- **Leakage controls:** rolling features use `.shift(1)`, and in-match statistics are removed as inputs.

---

## What's different about this approach

The coursework's goal was to explore new and unusual ways of tackling match prediction, not to squeeze out the highest accuracy with a conventional voting or averaging ensemble. The design choices below reflect that.

### Structure: base learners feed meta-learners

A typical ensemble has several models each predict an outcome, then combines them by voting or averaging. Here the two levels have different jobs:

1. **Base learners** (Poisson, CatBoost, XGBoost) do not make the final call. Each one outputs probabilities for Home, Draw and Away, which capture what that model has learned from the features.
2. **Meta-learners** (one per outcome) take those probabilities as input and learn how much to trust each base model for that outcome. Each outputs a probability for its own outcome.
3. **A softmax** over the three meta-learner outputs gives the final probabilities, and `argmax` gives the prediction.

Because the weighting is learned from out-of-fold predictions instead of being fixed, the system can rely on different models for different outcomes. For draws, it uses only XGBoost's probabilities.

### Design choices compared with a conventional approach

| Conventional approach | What we did instead | Why |
|---|---|---|
| Average or majority-vote several classifiers | Stack the base models through a learned meta-learner trained on out-of-fold predictions | The system learns how much to trust each model, per outcome, instead of using fixed equal weights |
| Static, season-level Poisson team strengths | Time-aware Poisson: match-specific attack and defence parameters from a rolling 5-match window | Captures current form while keeping a football-theory baseline |
| Models from a single family | A statistical model (Poisson) plus two boosted-tree models with different inductive biases | Diverse error profiles reduce correlated failures |
| Predict only the match outcome | An auxiliary P(Close) model predicts whether a match is decided by ≤ 1 goal, and its probability is fed to the classifiers as a learned feature | Close matches are more draw-prone, which static features struggle to express |
| One meta-learner for all classes | Three-part meta-learner (Home / Draw / Away). The draw path uses only XGBoost's probabilities, after coefficient analysis showed it over-weighted CatBoost | Each outcome can weight the models differently, and the draw signal came mainly from the draw-weighted model |
| Treat the ensemble as a black box | A linear (logistic regression) meta-learner | Coefficients can be inspected, which is how the draw-weighting problem was found |

### Auxiliary models

**P(Close), used in the final model.** A CatBoost binary classifier predicts whether a match will be decided by one goal or fewer, using only pre-match features. It outputs a probability instead of a class label: evenly matched teams (similar ELO ratings) get a high P(Close), and mismatches get a low one. Close matches are more draw-prone, and this is a pattern that static strength differences express poorly.

The probability is generated out-of-fold and appended as an input to the CatBoost and XGBoost classifiers. Unlike a hand-engineered feature, it is a learned representation: one model's prediction becomes another model's input.

**Other auxiliary targets, tried and dropped.** We also tried auxiliary models that predicted the number of fouls, yellow cards and red cards in a match, since discipline can strongly shape a result. The models could not predict these well from pre-match information, so they were not used. Those experiments are not included in this repository.

### Why not just use CatBoost?

In the notebook's final run, CatBoost alone is marginally ahead on accuracy (56.9% vs 56.1%) and Home F1, and macro F1 is tied (47.7%). We kept the stacked design because the aim was to investigate and document a different architecture and what it reveals, such as which models carry the draw signal and how leakage affects stacking, rather than to pick the single best-scoring model.

If the only goal were maximum accuracy, a tuned CatBoost would be the simpler choice and would score about as well.

---

## Results

Test set: 1,530 held-out matches, chronologically after all training and validation data.

| Metric | Value |
|---|---|
| Ensemble test accuracy | **56.2%** (report: 56.21%) |
| Bookmaker benchmark | ~53% |
| Always predict "Home" | ~46% |
| Ensemble draw recall | 13.7% (46 / 336) |
| XGBoost draw recall (1.3× weight) | 45.5% (153 / 336) |

Per-model results from the notebook's final run (small differences from the report figure are rerun variance):

| Model | Accuracy | F1 Home | F1 Draw | F1 Away |
|---|---|---|---|---|
| Poisson | 54.4% | 62.4 | 12.4 | 59.7 |
| CatBoost | 56.9% | 67.5 | 14.4 | 61.4 |
| XGBoost | 51.0% | 60.7 | 32.8 | 55.2 |
| **Ensemble** | **56.1%** | 67.1 | 14.3 | 61.8 |

**Reading the table:** the ensemble roughly matches the best single model (CatBoost) rather than clearly beating it. It does beat the bookmaker benchmark. See [Why not just use CatBoost?](#why-not-just-use-catboost) for why the stacked design was kept.

**Live check:** on 20 real fixtures from the 2025/26 season, played after the training cutoff, the model got 11 / 20 correct (55%). The sample is small and noisy, so treat it as a sanity check, not evidence.

Predictions for the 31 January 2026 fixtures are in [`predictions.csv`](predictions.csv).

---

## Key findings

**1. Random-shuffle validation inflated results.**
An early version split data with `shuffle=True`, letting future matches into training. Cross-validation looked strong, but the numbers were artefacts. Switching to strict chronological splits fixed this.

**2. Out-of-fold stacking improved generalisation.**
Training the meta-learner on out-of-fold predictions rather than in-sample predictions:

| | Without OOF | With OOF |
|---|---|---|
| Train accuracy | 58.10% | 48.75% |
| Test accuracy | 54.40% | **56.21%** |

Training accuracy fell and test accuracy rose, which is the signature of reduced overfitting.

**3. Diagnosing the draw problem.**
The meta-learner coefficients showed it over-weighting CatBoost's draw probability. Routing draw prediction through XGBoost only (the draw-weighted model) improved draw signal, but the ensemble's final draw recall stayed at 13.7%.

**4. ELO dominates.**
`elo_home_win_prob`, `elo_away_win_prob`, `elo_diff` and the raw ELO ratings rank highest in both CatBoost and XGBoost feature importance. External rating data mattered more than form features alone.

---

## Limitations

- **Draws are barely predicted.** Ensemble draw recall is 13.7% and the 20-match live check predicted zero draws against 7 actual. Draw boost shifts probabilities but rarely flips a confident Home/Away call.
- **Draw-weighted XGBoost trades accuracy for recall.** Draw recall reaches 45.5%, but overall accuracy falls to 51.0%.
- **No player-level data:** injuries, transfers and lineups are not modelled.
- **No betting odds** as features, so market information is not used.
- **Static features:** form is computed from historical data and is not refreshed closer to kick-off.
- **Small live sample:** 20 matches cannot distinguish skill from luck.

---

## Contributions

This was a group project.

I personally led the model innovation, training, validation and evaluation (70% contribution)


---

## Repository layout

```
.
├── README.md
├── epl_match_predictor.ipynb   # full pipeline, executed with outputs
├── architecture.png            # model architecture diagram
├── report.pdf                  # coursework report
└── predictions.csv             # predictions for 31 Jan 2026 fixtures
```

---

## Reproducing

1. Clone the repo and open `epl_match_predictor.ipynb` in Jupyter, VS Code or Colab.
2. Run all cells. The first code cell installs the required packages (numpy, pandas, scikit-learn, xgboost, catboost and others), so no separate setup is needed.

- The notebook loads training, test and enrichment data from published Google Sheet CSV links. See [Data sources](#data-sources).
- External enrichment (xG and similar) was originally fetched from an API and cached. The notebook ships with `fetched = True`, which loads the cached data instead of calling the API.
- To refetch, set `fetched = False` and supply your own API keys.
- Older experimental models in the notebook are guarded by `archived = True` and do not run by default.

---

## Data sources

Raw data is not committed to this repository.

| Data | Source |
|---|---|
| Historical matches (2000–2025) and results | Course-provided dataset (published Google Sheet) |
| Test fixtures (31 Jan 2026) | Course-provided dataset (published Google Sheet) |
| xG, possession and match statistics | External football statistics API (cached) |
| Team ELO ratings | [ClubElo.com](http://clubelo.com) |

---

## Notes on the notebook

- **Design goal:** the coursework aimed to explore novel architectures rather than maximise accuracy with a standard voting ensemble. This is why the stacked ensemble was kept even though CatBoost alone scores marginally higher. See [What's different about this approach](#whats-different-about-this-approach).
- Per-class figures vary slightly between runs because of model randomness.

---

## Acknowledgements

Coursework for COMP0036 (Machine Learning and Neural Computing), University College London. AI assistance is declared in the report under UCL's Category 2 policy.