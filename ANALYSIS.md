# Full Analysis: Predicting High-Cost Medicare Patients

This document contains the complete exploratory analysis, cost-burden segmentation deep dive, and predictive modeling results referenced in the [README](./README.md).

The analysis is organized around two stakeholder perspectives:

- **Health Analytics** — What does the Medicare population look like? What's associated with utilization and cost? Who are the highest-cost beneficiaries?
- **Payer Strategy** — Can we identify beneficiaries likely to become high-cost? Where might interventions be targeted, and why?

---

## 1. What Does the Medicare Population Look Like (2008–2010)?

### Population Demographics

The final dataset contains 332,694 patient-year records from 2008–2010. Because the unit of analysis is a patient-year, a beneficiary can contribute multiple records across different years.

**The dataset is disproportionately represented by White beneficiaries and women.** Women account for 55.57% of patient-year records versus 44.43% for men. By race:

| Group | Records | Share of sex |
|---|---|---|
| White women | 58,628 | 82.76% of female records |
| White men | 47,079 | 83.10% of male records |
| Black women | 7,682 | 10.80% of female records |
| Black men | 5,695 | 10.10% of male records |
| Hispanic women | 1,578 | 2.23% of female records |
| Hispanic men | 1,240 | 2.20% of male records |
| Other-race women | 2,950 | 4.16% of female records |
| Other-race men | 2,624 | 4.60% of male records |

This imbalance is an important limitation: patterns identified in this dataset may be more representative of the White Medicare population than of less-represented racial and ethnic groups.

**Most beneficiaries interact with multiple service types.** The average number of distinct service types accessed ranges from ~1.6 to 2.6 across demographic groups. Across the full population:

- 41.30% accessed 3 distinct service types
- 31.04% accessed 2 distinct service types
- 17.26% accessed 1 distinct service type
- 10.40% accessed all 4 service types

The lowest average service-type utilization appears among several of the oldest age groups and selected non-White populations (e.g., Black men 95+ contributed only 101 patient-year records; other-race men 95+ contributed 45). These small cells should be interpreted cautiously — lower utilization could reflect disease burden or access differences, but may also reflect small sample size.

### Service-Type Utilization

The largest utilization pattern is carrier + outpatient + prescription drug services (~37% of patient-year records), followed by carrier + prescription drug (~18.8%), then prescription-drug-only (~11.3%). Prescription drug and carrier/physician services are central to utilization overall, while inpatient services are used by a smaller subset.

### Utilization by Age

Utilization patterns are broadly consistent across age bands, with carrier and prescription drug services accounting for the largest shares. One notable exception: the **under-65 Medicare population** (who generally qualify via disability or specific conditions) shows relatively high prescription drug utilization, accompanied by higher drug-related payer costs — a subgroup worth investigating further when examining cost drivers.

### Cost by Age

Although service-type distribution is relatively stable across age, **payer costs increase with age for several categories**:

- Inpatient payer costs increase with age, peaking at 85–94
- Carrier payer costs increase with age, peaking at 85–94
- Prescription drug costs generally increase with age
- Other categories remain comparatively stable

This divergence between stable utilization and rising cost suggests utilization volume alone doesn't fully explain spending — increasing clinical complexity and care intensity among older beneficiaries likely contributes even when the number of distinct service types stays similar. This motivates the chronic-condition and high-cost analysis below.

---

## 2. Who Are the High-Cost Beneficiaries?

High-cost status was defined independently per year using the top 10% of annual payer/OOP spending, to account for year-to-year variation in spending distributions:

| Year | Top 10% Payer Cost Cutoff | Top 10% OOP Cost Cutoff |
|---|---|---|
| 2008 | $12,520 | $2,544 |
| 2009 | $13,530 | $2,578 |
| 2010 | $7,340 | $1,590 |

### ESRD Is Strongly Associated With High Healthcare Costs

ESRD represents ~7.86% of patient-year records but is disproportionately represented among high-cost beneficiaries: 27.3% of high payer-cost records and 28.63% of high OOP-cost records are ESRD, despite ESRD's much smaller overall share. ESRD beneficiaries have roughly 3x the average payer and OOP cost of non-ESRD beneficiaries, alongside higher utilization — making ESRD an important clinical indicator for payer strategy.

### The 2010 Cost Decline Requires Further Investigation

Costs declined sharply in 2010 for the same ~110,898-beneficiary cohort. This is treated as an observation, not a causal conclusion — possible explanations include changes in utilization, claims patterns, reimbursement, or coding. External research (Commonwealth Fund) found Medicare spending growth slowed over this broader period, partly attributed to declining cardiovascular disease spending, but this dataset alone (a specific DE-SynPUF sample, 2008–2010 only) cannot attribute the decline to specific policy causes like the ACA.

### Cost-Burden Profiles Across Population Subgroups

Patient-year records were grouped into four cost-burden profiles based on whether >10% of a subgroup's records fell into the high payer-cost and/or high OOP-cost categories:

| Profile | Interpretation | Example subgroups |
|---|---|---|
| 🟧 High OOP burden, lower payer spending | High OOP prevalence, lower payer prevalence | Hispanic females 75–84; Hispanic males 85–94 |
| 🟨 High payer spending, lower OOP burden | High payer prevalence, lower OOP prevalence | Black males 95+ |
| 🟥 High prevalence of both | High prevalence of both | Females 85–94 and <65; non-Hispanic males 85–94; non-Black males 95+ |
| 🟩 Low prevalence of both | Low prevalence of both | Ages 65–74 across demographic groups |

The high-burden-both profile concentrates among females 85–94 and <65, and males 85–94 and 95+ — suggesting these age groups face substantial financial burden from both payer and patient perspectives. Notably, the Black male 95+ (high-payer/low-OOP) and non-Black male 95+ (high-both) subgroups both show relatively low average service-type utilization, suggesting distinct-service-type count alone doesn't capture cost intensity — a beneficiary can interact with fewer service types while still generating substantially higher costs. The low-burden-both profile concentrating at 65–74 is consistent with the lower service-type utilization observed in that group earlier.

### Cost Composition by Cost-Burden Profile

- **Inpatient & outpatient payer costs**: highest among the high-burden-both group, followed by the high-payer/lower-OOP group
- **Carrier payer costs**: highest for high-burden-both; high-OOP/lower-payer and low-burden-both are nearly tied for second
- **Prescription drug (PDE) payer costs**: diverges from the other patterns — the high-OOP/lower-payer group has the *highest* average PDE payer cost, exceeding the high-burden-both group by ~$500

This divergence is notable because the high-OOP/lower-payer group is defined by substantial patient burden despite lower payer-spending prevalence — suggesting prescription-related expenses may contribute disproportionately to their OOP burden relative to other service categories. Distinguishing between higher drug utilization vs. greater cost-sharing would require further analysis of prescription volume and patient cost-sharing structure.

**Key takeaway:** High-cost Medicare beneficiaries are not a homogeneous population. The high-burden-both group is the most consistently expensive across inpatient/outpatient/carrier services, while the high-OOP/lower-payer group stands out for prescription drug costs specifically — meaning intervention strategy may need to differ depending on whether the primary burden is medical utilization, drug spending, or cost-sharing.

### Utilization vs. Cost

The utilization-to-cost relationship varies by service type:

- **Inpatient**: follows the expected pattern (higher utilization → higher cost), but the gap in utilization between the two highest-utilization groups is small relative to their ~$300 cost gap — utilization volume alone doesn't fully explain the difference
- **Outpatient**: less consistent — the high-OOP/lower-payer and low-burden-both groups show *higher* utilization than the high-payer/lower-OOP group despite *lower* payer costs, suggesting outpatient cost depends more on encounter intensity/composition than volume
- **Carrier & PDE**: much more direct relationship — higher utilization corresponds with higher payer cost

Outpatient claims are flagged as a useful next area for analysis (cost per utilization event, procedure/provider mix).

### Disease Burden and Cost-Burden Profiles

Across six chronic conditions (diabetes, cancer, CKD, COPD, ESRD, heart failure), a consistent ordering emerges:

**High-burden-both → High OOP/lower payer → Low burden-both → High payer/lower OOP**

COPD illustrates this clearly: the high-burden-both group has the highest prevalence, followed by high-OOP/lower-payer; the high-payer/lower-OOP group consistently has the *lowest* disease prevalence. The same ordering holds across all six conditions.

**Key insight:** Clinical disease burden appears more closely associated with patient OOP burden than with payer-only cost burden in this dataset — possibly because chronic/complex conditions require sustained treatment across multiple services, generating recurring cost-sharing even without individual high-cost claims. This suggests high payer spending and high OOP burden may represent different risk phenotypes: payer-heavy populations may be driven by a smaller number of high-intensity services, while disease-heavy populations experience more persistent needs and associated OOP expense. The current data can't establish causal direction between disease burden and OOP spending.

**Suggested future analysis:** claims by disease, cost per utilization event by condition, chronic condition count per patient-year, service-mix by disease, and whether disease-specific utilization predicts OOP burden after controlling for age/demographics.

---

## 3. Can We Identify Future High-Cost Beneficiaries?

Models were trained to predict high payer-cost and high OOP-burden outcomes, using progressively richer historical features:

- **2008**: baseline demographics + ESRD status only
- **2009**: adds 2008 historical utilization/spending
- **2010**: adds cumulative 2008–2009 history

High-cost outcomes were defined per-year using top-10% thresholds (payer cost and OOP burden prevalence both landed at ~9.97–10.00% across years, as expected from the threshold definition).

### Model Comparison

**Logistic Regression** (balanced class weights, high-recall screening):

| Target | ROC-AUC | Precision | Recall |
|---|---|---|---|
| High Payer Cost 2008 | 0.66 | 0.25 | 0.37 |
| High OOP Burden 2008 | 0.67 | 0.27 | 0.38 |
| High Payer Cost 2009 | 0.83 | 0.24 | 0.75 |
| High OOP Burden 2009 | 0.89 | 0.32 | 0.83 |
| High Payer Cost 2010 | 0.77 | 0.19 | 0.72 |
| High OOP Burden 2010 | 0.81 | 0.22 | 0.77 |

Performance improved substantially once historical features were available (2009/2010 vs. 2008), especially recall. Low precision means many flagged beneficiaries won't end up high-cost — better suited to broad screening than selective intervention.

**Random Forest** (intermediate precision/recall tradeoff):

| Target | ROC-AUC | Precision | Recall |
|---|---|---|---|
| High Payer Cost 2008 | 0.69 | 0.22 | 0.43 |
| High OOP Burden 2008 | 0.70 | 0.24 | 0.46 |
| High Payer Cost 2009 | 0.83 | 0.37 | 0.49 |
| High OOP Burden 2009 | 0.90 | 0.43 | 0.68 |
| High Payer Cost 2010 | 0.79 | 0.32 | 0.38 |
| High OOP Burden 2010 | 0.82 | 0.31 | 0.48 |

Generally trades some recall for improved precision relative to logistic regression — a useful middle ground when false positives are costlier.

**XGBoost** (strongest discriminative performance; used for SHAP interpretation):

| Target | ROC-AUC | Precision | Recall |
|---|---|---|---|
| High Payer Cost 2008 | 0.69 | 0.48 | 0.03 |
| High OOP Burden 2008 | 0.71 | 0.61 | 0.03 |
| High Payer Cost 2009 | 0.84 | 0.75 | 0.15 |
| High OOP Burden 2009 | 0.90 | 0.67 | 0.33 |
| High Payer Cost 2010 | 0.79 | 0.75 | 0.07 |
| High OOP Burden 2010 | 0.83 | 0.68 | 0.08 |

XGBoost's high precision means positive predictions are relatively high-confidence, but low recall means it captures only a small fraction of the true high-cost population at the default threshold — strong discriminative performance (ROC-AUC) doesn't automatically translate to high-recall targeting at a fixed threshold.

### What Drives High-Cost Predictions?

Feature-family analysis across models consistently pointed to historical cost, utilization, disease burden, chronic condition burden, and demographics as important predictive information. XGBoost SHAP analysis identified the strongest individual contributors:

- Number of carrier (Part B) claims
- Total carrier payer expenditures
- Number of outpatient claims
- Prescription drug expenditures
- Outpatient payer expenditures

These reinforce the EDA findings: carrier and prescription drug utilization have a fairly direct relationship with payer cost, while outpatient claim count shows nonlinear effects in the model — consistent with the earlier finding that outpatient utilization doesn't map proportionally to payer cost.

**Clinical/coverage indicators:** ESRD consistently emerged as an important high-cost risk indicator, reinforcing the EDA finding on ESRD cost burden. Partial-year Medicare coverage was also associated with higher predicted future cost (possibly new enrollment or coverage transitions — a hypothesis for further investigation, not a established causal mechanism).

### Risk Stratification

Predicted probabilities were used to rank beneficiaries into five equal-sized risk quintiles (Q1 lowest 20% risk → Q5 highest 20%). Event rates increased monotonically from Q1 to Q5 across models. For example, in the 2008 High OOP Burden model, observed event rate rose from 5.3% (Q1) to 21.4% (Q5) — a 4.04x difference — with mean predicted probability rising from 30.4% to 64.0%.

Treating the top quintile (Q5) as an intervention population achieved **25.5% precision and 51.1% recall** — targeting the top 20% of predicted-risk beneficiaries captured about half of the actual high-cost population, with roughly 1 in 4 targeted beneficiaries ultimately high-cost. This represents meaningful risk concentration, useful for prioritizing beneficiaries for review in a low-cost-screening context, though a substantial share of high-cost beneficiaries remain outside the top quintile — more resource-intensive interventions would likely need additional predictors or more targeted thresholds.

### Overall Modeling Takeaway

Future high-cost risk is partially predictable from demographics, prior utilization, spending, and clinical burden — particularly once historical information is available. There's an important distinction between risk *ranking* and high-confidence *classification*: XGBoost offers strong discrimination and interpretability, while logistic regression offers substantially higher recall at its selected threshold, better suited to broad screening.

More broadly, both the EDA and modeling results point to the same conclusion: **high-cost healthcare utilization is driven by a combination of clinical complexity, historical utilization, and spending patterns — not by utilization volume alone.** Claims-based risk stratification looks like a viable first-stage screening tool, but richer longitudinal information and more detailed service-intensity features would likely be needed to support higher-stakes, more selective intervention targeting.
