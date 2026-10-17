# Dataset notes

## Selected public-use data

- Source: [CDC/NCHS 2023 NHIS documentation](https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html), Sample Adult interview, Public use data section.
- File: `data/raw/adult23csv.zip` (downloaded 2026-10-17); extracted files: `data/raw/adult23csv/adult23.csv` and `readme.txt`.
- Content: 29,522 sample-adult records and 647 columns. The CDC readme warns that CSV readers should infer data types from enough or all rows to avoid truncating sparse or multi-digit values.
- Variables for the proposed proxy outcome: `CANEV_A` (ever diagnosed with cancer) and `PHSTAT_A` (self-rated health). In the cancer-history subset, define a provisional binary outcome as fair/poor (`PHSTAT_A` code 4 or 5) versus excellent/very good/good (codes 1–3); verify codes against the [CDC adult codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/adult-codebook.pdf) before analysis.
- Access: direct public-use download; no account or guestbook required. Raw data is excluded by the root `.gitignore` and should remain out of version control.

## Scope and limitations

This dataset does **not** record chemotherapy exposure, PROMIS physical-health change, or a pre-treatment/post-treatment sequence. It supports only a cross-sectional classification task among adults reporting a cancer diagnosis. That is a substantial proxy/reframing of the assigned problem, not a reproduction of the Ostberg–Peterson study. Confirm with the TA that this target is acceptable before modeling. Avoid describing results as predicting chemotherapy outcomes or decline.

The reference paper's 1,252-patient cohort came from Stanford's STARR clinical data repository and is not distributed with the report. ICPSR 39628 was checked as another candidate, but the downloaded package contained documentation/reports rather than patient-level data. Harvard Dataverse's PROMIS 2 MY Health data required submission of personal guestbook details, so it was not downloaded.

## Next data step

Check value codes against the codebook, subset `CANEV_A` respondents, then report missingness and label balance for `PHSTAT_A`. Use survey weights only if the project scope and methods support a survey-aware analysis; ordinary ML splits may not produce population-representative estimates.
