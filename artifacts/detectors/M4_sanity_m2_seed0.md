# M-4 — Detector pipeline sanity check (M2)

**Milestone:** M-4 (SPEC §12) · **Commit:** `26e2492cc792aa059fe1da8aa21b1a2ff56305fe-dirty` · **Generated:** 2026-09-20T16:45:26+0600

> **This page produces no thesis number.** M-4 is a pipeline sanity gate on M2, the
> smoke-test model (SPEC §3). The first reported detector numbers come from M-5 on M2.

## Verdict

**M-4 sanity NOT MET.** 1 of 15 calibrated FPRs fall outside 0.3×–3.0× of nominal — the pipeline is miscalibrated; fix before M-5.

## Calibration and detection

| detector | alpha | tau | n | alarms | fpr | fpr_upper95 | fpr_ratio | tpr_H | tpr_S | precision | tap_set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.0500 | 26.4928 | 2000 | 119 | 0.0595 | 0.0689 | 1.1900 | 0.0982 | 0.0523 | int8 | default |
| D1 | 0.0100 | 78.8479 | 2000 | 22 | 0.0110 | 0.0157 | 1.1000 | 0.0060 | 0.0100 | int8 | default |
| D1 | 0.0010 | 217.1128 | 2000 | 2 | 0.0010 | 0.0031 | 1.0000 | 0.0000 | 0.0009 | int8 | default |
| D2 | 0.0500 | -0.1279 | 2000 | 122 | 0.0610 | 0.0705 | 1.2200 | 0.2099 | 0.0497 | int8 | default |
| D2 | 0.0100 | -0.0789 | 2000 | 24 | 0.0120 | 0.0168 | 1.2000 | 0.0345 | 0.0102 | int8 | default |
| D2 | 0.0010 | 0.0143 | 2000 | 3 | 0.0015 | 0.0039 | 1.5000 | 0.0045 | 0.0013 | int8 | default |
| D3_max | 0.0500 | 0.9885 | 2000 | 110 | 0.0550 | 0.0641 | 1.1000 | 0.1034 | 0.0547 | int8 | default |
| D3_max | 0.0100 | 0.9980 | 2000 | 29 | 0.0145 | 0.0197 | 1.4500 | 0.0337 | 0.0114 | int8 | default |
| D3_max | 0.0010 | 1.0000 | 2000 | 0 | 0.0000 | 0.0015 | 0.0000 | 0.0000 | 0.0000 | int8 | default |
| D3_mean | 0.0500 | 0.7856 | 2000 | 94 | 0.0470 | 0.0555 | 0.9400 | 0.1297 | 0.0405 | int8 | default |
| D3_mean | 0.0100 | 0.8869 | 2000 | 22 | 0.0110 | 0.0157 | 1.1000 | 0.0210 | 0.0079 | int8 | default |
| D3_mean | 0.0010 | 0.9593 | 2000 | 5 | 0.0025 | 0.0052 | 2.5000 | 0.0052 | 0.0011 | int8 | default |
| D3_fpr_weighted | 0.0500 | 0.7856 | 2000 | 94 | 0.0470 | 0.0555 | 0.9400 | 0.1297 | 0.0405 | int8 | default |
| D3_fpr_weighted | 0.0100 | 0.8869 | 2000 | 22 | 0.0110 | 0.0157 | 1.1000 | 0.0210 | 0.0079 | int8 | default |
| D3_fpr_weighted | 0.0010 | 0.9593 | 2000 | 5 | 0.0025 | 0.0052 | 2.5000 | 0.0052 | 0.0011 | int8 | default |

`alpha` is the nominal FPR fixed a priori (§8, C3); `fpr` is measured on `clean_test`;
`fpr_upper95` is the Clopper-Pearson bound, which is what 0.1 % means on this many clean samples.
TPR columns are per track and never merged (§4.3).

## Fitted detectors

| model | precision | tap_set | detector_kb | fault_features |
|---|---|---|---|---|
| m2 | int8 | default | 20314.200 | 144000 |

Dropped features (IQR below the floor on `clean_fit`, §5.3): 35.
