# Step 1: Dataset and proposed scope

## Dataset

Downloaded the CDC 2023 NHIS Sample Adult public-use dataset. The ZIP and extracted CSV are in `data/raw/`; this directory is excluded from Git. The CSV contains 29,522 records and 647 columns. Source and data notes: [CDC 2023 NHIS documentation](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html) and [`data/README.md`](../data/README.md).

## Proposed project target

Classify fair/poor self-rated health among adults who report a prior cancer diagnosis, using `PHSTAT_A` and `CANEV_A` after confirming their codes in the CDC codebook.

## Scope caveat

The assigned paper studies patient-reported health decline for cancer patients on chemotherapy. NHIS is cross-sectional and does not contain a chemotherapy variable or repeated health measure. This target is a proxy and requires TA approval before modeling. The report must not present it as predicting chemotherapy outcomes or decline.

## Next step

Confirm the proxy target with the TA. Then inspect the relevant codebook values, missingness, and label balance in the cancer-history subset before choosing features or a model.
