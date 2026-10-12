# Implementation note 002 — parent classifier alignment

**Recorded:** 2026-09-30, before classifier execution or downstream outcomes.

A pre-execution protocol audit found that the draft downstream script specified
the SAGA solver, while the frozen plan requires the classifier protocol to match
the curation parent. The script now uses the parent's standardized multinomial
linear head: scikit-learn logistic regression with `C=1`, the `lbfgs` solver,
and a 500-iteration cap. Arms retain identical inputs, caps, and convergence
rules, and realized iteration counts remain in the raw run table.

No selection policy, feature, arm, threshold, outcome, or statistical rule was
changed. No classifier or test outcome existed when this correction was made.
