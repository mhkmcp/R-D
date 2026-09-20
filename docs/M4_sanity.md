# M-4 — Detector pipeline sanity check (M1)

**Milestone:** M-4 (SPEC §12) · **Commit:** `26e2492cc792aa059fe1da8aa21b1a2ff56305fe-dirty` · **Generated:** 2026-09-20T16:24:05+0600

> **This page produces no thesis number.** M-4 is a pipeline sanity gate on M1, the
> smoke-test model (SPEC §3). The first reported detector numbers come from M-5 on M2.

## Verdict

**M-4 sanity NOT MET.** 5 of 15 calibrated FPRs fall outside 0.3×–3.0× of nominal — the pipeline is miscalibrated; fix before M-5.

No fault features were present, so only the calibration half of the gate ran. Generate them with `fidnn inject --tap-set …`.

## Calibration and detection

| detector | alpha | tau | n | alarms | fpr | fpr_upper95 | fpr_ratio | precision | tap_set |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.0500 | 4.2739 | 400 | 19 | 0.0475 | 0.0689 | 0.9500 | fp32 | default |
| D1 | 0.0100 | 14.1308 | 400 | 7 | 0.0175 | 0.0326 | 1.7500 | fp32 | default |
| D1 | 0.0010 | 35.9003 | 400 | 3 | 0.0075 | 0.0193 | 7.5000 | fp32 | default |
| D2 | 0.0500 | 0.2614 | 400 | 17 | 0.0425 | 0.0631 | 0.8500 | fp32 | default |
| D2 | 0.0100 | 9.3185 | 400 | 3 | 0.0075 | 0.0193 | 0.7500 | fp32 | default |
| D2 | 0.0010 | 14.7408 | 400 | 2 | 0.0050 | 0.0157 | 5.0000 | fp32 | default |
| D3_max | 0.0500 | 0.9900 | 400 | 18 | 0.0450 | 0.0660 | 0.9000 | fp32 | default |
| D3_max | 0.0100 | 0.9975 | 400 | 10 | 0.0250 | 0.0420 | 2.5000 | fp32 | default |
| D3_max | 0.0010 | 1.0000 | 400 | 0 | 0.0000 | 0.0075 | 0.0000 | fp32 | default |
| D3_mean | 0.0500 | 0.7703 | 400 | 25 | 0.0625 | 0.0862 | 1.2500 | fp32 | default |
| D3_mean | 0.0100 | 0.8280 | 400 | 12 | 0.0300 | 0.0482 | 3.0000 | fp32 | default |
| D3_mean | 0.0010 | 0.8933 | 400 | 2 | 0.0050 | 0.0157 | 5.0000 | fp32 | default |
| D3_fpr_weighted | 0.0500 | 0.7703 | 400 | 25 | 0.0625 | 0.0862 | 1.2500 | fp32 | default |
| D3_fpr_weighted | 0.0100 | 0.8280 | 400 | 12 | 0.0300 | 0.0482 | 3.0000 | fp32 | default |
| D3_fpr_weighted | 0.0010 | 0.8933 | 400 | 2 | 0.0050 | 0.0157 | 5.0000 | fp32 | default |

`alpha` is the nominal FPR fixed a priori (§8, C3); `fpr` is measured on `clean_test`;
`fpr_upper95` is the Clopper-Pearson bound, which is what 0.1 % means on this many clean samples.


## Fitted detectors

| model | precision | tap_set | detector_kb | fault_features |
|---|---|---|---|---|
| m1 | fp32 | default | 4959.900 | None |

Dropped features (IQR below the floor on `clean_fit`, §5.3): 42.
