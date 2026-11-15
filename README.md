# Bayesian marketing mix model for a B2C subscription business

**Question:** which channels drive the value of new subscribers, and should the media budget move?

A Bayesian marketing mix model (MMM) in PyMC for a multi-market subscription product spending about $75k a week on media. The data is simulated, so every answer can be checked against the truth.

## Key findings

1. **The model reads the past well.** It forecasts 12 unseen weeks within 5.9%, and the true return of every channel falls inside its estimated range.
2. **Acting on its best guess would lose money.** A budget optimiser that trusts the model's central estimate moves money out of paid social and loses $3.1k of pRevenue a week, about $160k a year.
3. **The weak point is the next dollar, not the average one.** The model knows roughly what each channel has returned. It does not know what an extra dollar would return, and that is what a budget decision depends on.
4. **Two experiments plus a cautious decision rule fix it.** With two lift tests in the model, its best plan turns a small real gain. The cautious rule declines to move budget on what is close to a coin flip.

## Business implications

- **Use the MMM to decide where to test, and experiments to decide where to spend.** The model is reliable on direction and on average returns, but not on marginal ones.
- **Reallocating a fixed budget is worth little here, and getting it wrong costs more.** The best possible reshuffle adds 3.3% to media-driven pRevenue; the naive plan loses 2.4%.
- **Placing new budget well is worth more.** With 20% more budget, the calibrated model places the extra money for $19.3k a week of pRevenue, against $13.3k from the uncalibrated model: about $315k a year from the same spend.
- **Design tests around the decision.** To decide whether to scale a channel, test scaling it. Switching it off only shows its average return.

## The business

A subscription product sold in three groups of markets, priced in USD, simulated weekly from January 2023 to December 2025.

| | |
|---|---|
| New subscribers | about 395,000 in total, 1,900 to 3,300 a week |
| Markets | three price tiers (tier 2 pays 75% and tier 3 pays 50% of the tier 1 price) |
| Paid media | search, paid social, display, influencer, CTV (connected TV, run in flights) |
| Owned media | email, measured as accounts reached |
| Media budget | about $75k a week on average; CTV flights push some weeks above $100k |
| Other drivers | January and September peaks, a late-December low, Black Friday promos, three product launches, two price rises |

![Simulated data](artifacts/data_overview.png)

*Weekly pRevenue rises with each product launch and dips every December. Influencer budgets follow the same seasonal pattern as search (82% correlated), which makes the two hard to separate, as in most real media plans.*

## What the model explains: pRevenue

**pRevenue** is the predicted 24-month revenue of each week's new subscribers. A separate CLV model (customer lifetime value) scores each subscriber on plan, market tier, device, payment method, age band, sale hook (the offer that converted them), promo discount and acquisition channel. Earlier CLV analysis found that channel adds little once the other attributes are known; that is re-checked as data grows and would feed back into the MMM if it changed.

pRevenue rewards channels that bring valuable subscribers, not just many, and is known the week someone joins. A second model on new-subscriber counts runs alongside to compare volume with value.

## How the model works

Weekly pRevenue = baseline + the effect of each channel + noise.

- **Baseline:** trend, seasonality, promo weeks, product launches and price.
- **Carryover (adstock):** part of this week's spend still works next week. CTV carries over longest.
- **Diminishing returns (saturation):** each extra dollar in a channel buys a little less than the one before.
- **Priors on return per dollar:** a prior is the model's starting assumption. Each paid channel starts from "a dollar usually returns $0.5 to $4.7 of pRevenue", a belief a marketer can challenge.
- **Lift tests (optional):** an experiment result, such as "raising social spend by 50% added $15k a week", enters as a measured point on that channel's response curve.

## Does it work?

![Model fit](artifacts/p_revenue/fit.png)

*Fitted pRevenue tracks the actual series closely (3.2% error in training, 5.9% on the 12 held-out weeks). The forecast misses the Black Friday spike and the December low after it: the model treats a promo as a fixed bump, although its real effect grows with demand.*

![Estimated vs true](artifacts/p_revenue/recovery.png)

*Every true value sits inside the model's 90% range. Display and influencer are overrated: display is too small to measure, and influencer moves with search. Social is underrated, because its slow budget drift looks like trend. Email gets 13% of the credit against a true 6.5%, because it peaks in the same weeks as seasonal demand.*

## From model to budget

The optimiser reshapes a fixed weekly budget across search, social, display and influencer. CTV is held, because it is bought in flights rather than at a steady weekly level. No channel moves more than 50% from today's spend, the range the model has seen.

It compares two rules:
- **Expected value:** follow the model's central estimate.
- **Cautious:** move budget only where the gain still holds if the model is somewhat wrong (expected gain minus one standard deviation).

![Marginal returns](artifacts/p_revenue/marginal_returns.png)

*What the next dollar returns at today's spend. The model puts social at 0.6 against a true 1.6, so it treats the channel with the most headroom as nearly exhausted. Average returns were recovered well; these marginal returns were not.*

![Budget plans without lift tests](artifacts/p_revenue/budget_plan.png)

*Without experiments, the expected-value plan cuts social by 36% and loses $3.1k a week in reality. The cautious plan barely moves, which is the right instinct.*

The true outcome of each plan, against media-driven pRevenue of about $127k a week:

| Model | Expected-value plan | Cautious plan |
|---|---|---|
| Without lift tests | −$3.1k a week (−2.4%) | −$0.2k a week (about even) |
| With two lift tests | +$1.5k a week (+1.2%) | holds today's split |
| With two lift tests, budget +20% | +$19.3k a week | +$18.5k a week |
| Best possible reshuffle | +$4.2k a week (+3.3%) | |

![Budget plans with lift tests](artifacts/p_revenue_calibrated/budget_plan.png)

*With a social scale-up test and an influencer scale-down test, the expected-value plan trims display, the weakest channel, and gains in reality. The model still gives it only a 52% chance of beating today, so the cautious rule holds. Social remains underrated (0.9 against 1.6); a search test would be the next experiment.*

The new-subscriber model tells the same story. Volume and value rank channels almost identically, consistent with channel having little effect on subscriber value.

## Limitations

- The data is simulated. Its true process differs from the model's assumptions, but less than real data would.
- The lift-test results in `data/lift_tests.json` are placeholders set close to the true values. A follow-up project on geo-test design will generate them from a simulated experiment.
- One model covers all markets. Promo effects are fixed, and CTV is evaluated on its average weekly spend rather than by flight.

## Next steps with real data

1. **Run the two experiments for real:** a 50% scale-up of social and a scale-down of influencer, in matched regions, for at least six weeks. Then a search test, where the model is least sure.
2. **Split the model by market tier,** sharing information across tiers (a hierarchical model), so each tier gets its own response curves.
3. **Re-check the pRevenue assumptions** every quarter: whether channel still has little effect on value, and whether CLV model updates shift the series. Re-score history after any model change.
4. **Refit monthly and track the forecast error** on new weeks, with an alert if it drifts above 8%.
5. **Model CTV by flight,** to plan flight timing and length rather than a weekly average.

---

<details>
<summary>Repository and how to run</summary>

```
src/mmm/
  simulate.py     data generator, with ground truth
  transforms.py   carryover and diminishing returns
  data.py         model inputs
  model.py        PyMC model, with return-per-dollar priors and lift tests
  optimize.py     response curves, marginal returns, budget optimiser
  truth.py        true response curves, for evaluation only
scripts/
  fit_model.py        fit, check, compare with truth
  optimize_budget.py  budget plans and their true outcome
  plot_data.py        overview of the simulated data
data/             weekly data, subscriber data, lift tests, truth
artifacts/        charts, tables and saved model results
tests/            unit tests
```

Open in GitHub Codespaces (the dev container installs everything), or use Python 3.12:

```
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh    # data, tests, four model fits, budget plans: about 20 minutes
```

Numbers can vary slightly between machines.
</details>

Built by Milesh Vadodaria, with AI assistance for code. Modelling choices, checks and conclusions are my own.
