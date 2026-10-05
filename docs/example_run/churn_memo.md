# Month-to-month, early-tenure customers with repeated support calls are most likely to churn; retention should focus on them

**Bottom line:** Churn risk is concentrated among month-to-month customers, especially in their early months and after repeated support contacts. Month-to-month customers carry an adjusted odds ratio of 4.70 for churn versus two-year contracts [F27]. Use the logistic-regression score to rank customers for outreach rather than to make hard predictions, because churn is only moderately predictable (ROC-AUC 0.7767) [F15]. Before scaling, confirm with the data owner how tenure and support calls are timed relative to churn [F22].

## Key findings
- **Contract type is the strongest marker.**
  - Churn is 34.4% for month-to-month customers, versus 12.8% for one-year and 12.2% for two-year contracts [F9]. The base rate is 0.246 [F18].
  - The gap holds within every tenure band. Among customers in their first 1-12 months, churn is 57.2% for month-to-month, versus 23.8% for one-year and 19.9% for two-year [F26].
  - After adjusting for tenure, support calls and charges, one-year contracts are not distinguishable from two-year (OR 1.14, CI 0.85-1.53) [F27]. The risk is specific to month-to-month, not to all non-two-year contracts.
- **Shorter tenure is associated with higher churn.**
  - Churners average 26.6 months of tenure, versus 39.2 for stayers (Hedges g -0.62) [F10].
  - Churn is 0.607 at tenure 1, versus 0.089 at tenure 72 [F3].
  - The association holds within each contract type (Spearman rho -0.30, -0.25 and -0.15), so it is not just contract confounding [F11].
- **Repeated support calls are associated with higher churn, independent of contract and tenure.**
  - Overall, churn rises from 16.2% at 0 calls to 50% at 5 calls [F12].
  - Among month-to-month customers, churn rises from 22.8% at 0 calls to 50.6% at 3 calls and 55.6% at 5+ calls [F25].
  - After joint adjustment, each additional call has an OR of 1.49 (CI 1.40-1.59) [F28].
- **Monthly charges play a minor role.** Churners pay a mean of 72.8 versus 68.8 for stayers, a small effect (Hedges g 0.205) [F13].
- **The model ranks customers moderately well.**
  - Logistic regression reaches accuracy of 0.7863, versus a majority baseline of 0.754 [F15]. Random forest (0.7403) and XGBoost (0.7114) did no better [F15].
  - By mean |SHAP|, the top drivers are contract (0.648), tenure_months (0.620) and support_calls (0.412) [F16].

## Recommended actions
1. **Target month-to-month customers first.** Offer incentives to move them to longer contracts, since even one-year contracts behave like two-year [F9, F27].
2. **Run an early-tenure retention program.** Focus on new month-to-month customers, where churn is highest [F26, F11].
3. **Trigger proactive outreach after 3+ support calls.** This trigger is most useful for month-to-month and early-tenure customers [F25]. Hold it pending confirmation that cancellation calls are not counted [F22, F28].
4. **Rank the outreach list with the logistic-regression score.** Expect many misses; do not use the score as a yes/no prediction [F6, F15].
5. **Score customers offline before finalising the list.** Top-decile overlap and precision could not be computed with the current tools [F33].
6. **Send the data owner four questions:** the snapshot date, how tenure is computed for churners, the support_calls window, and whether calls are logged before or after the cancellation request [F22].
7. **Do not lead with price-based targeting.** Charges are a minor secondary factor [F13].

## What we ruled out
- **Payment method:** no association with churn (Cramér's V 0.000) [F14, E11].
- **Signup channel:** no association with churn (V 0.000) [F14, E12].
- **Region:** statistically significant but trivial (V 0.039) [F14, E10].

Do not segment retention efforts by any of these three.

## Caveats
- **Observational data.** All results are associations, not causal effects. Moving customers to longer contracts is not guaranteed to reduce churn.
- **Possible leakage in tenure and support calls.** The data has no date columns, so we cannot verify whether tenure and support calls were recorded before churn [F22].
  - Without support_calls, ROC-AUC falls only to 0.7588 [F17].
  - Without tenure_months, ROC-AUC falls to 0.7171 and accuracy to 0.7538, which is at the baseline [F31].
  - Without both, ROC-AUC is 0.6766, leaving only a coarse month-to-month focus [F32].
- **Support calls appear to be a window-limited count, not a lifetime total.** Churners average 1.923 calls versus 1.408 for stayers in every tenure band [F20].
- **Small cells at high call counts.** The 5+ call cells are small (n=19-37 per stratum), so read those rates cautiously [F25].
- **Data quality is otherwise good.** All 4000 customer IDs are unique, and there are no duplicates or missing values [F1]. Categorical levels are clean [F5].