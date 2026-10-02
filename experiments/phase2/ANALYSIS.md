# Analysis and handoff plan (before phase-two test evaluation)

The 45 training attempts produce 33 newly selected models for test evaluation:
6 fixed-recipe full-data replications, 9 random frozen controls, and 18 selected
equal-search models (three families × two fractions × three seeds).

Reported tables have 45 rows across comparison groups because phase-one results
are explicitly reused and full-data pretrained probes appear in two comparisons:
fixed_full=9, frozen=18, equal_search=18. Do not call these 45 independent tests.

Planned patient-paired bootstrap contrasts: pretrained minus scratch for each
fixed-full seed (3 pairs); pretrained frozen minus random frozen at each matched
fraction/seed (9 pairs); pretrained minus scratch, xResNet minus scratch, and
pretrained minus xResNet for each equal-search fraction/seed (18 pairs). Each
pair gets 2,000 patient bootstrap replicates and percentile 95% intervals for
AUROC and average precision. These 30 pairwise intervals are descriptive and
unadjusted; no familywise significance claim, seed-pooled pseudo-replication,
or equivalence/noninferiority claim will be made.

Before test access, verify each new frozen model against full-waveform validation
inference, verify all original phase-one artifacts, reconstruct selection from
all candidate validation scores, and write a separate test lock. Independent
rank-AUROC, tie-aware AP and confusion-count checks validate exported metrics.
Show all candidate validation scores, cap hits, exposures and numerical failures.

Output: results/phase2/{metrics,per_class,summary,training_audit,bootstrap}.csv,
results/phase2/FINDINGS.md and aggregate figures under figures/phase2/.
Never overwrite phase-one results or replace its single-seed 100% result silently.
