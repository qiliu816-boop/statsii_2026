# Replication and Extension of *The Development of an Issue Public: Evidence from The Eras Tour*

**Author:** Qi Liu  
**Course:** Applied Statistical Analysis II  
**Programme:** MSc Applied Social Data Science, Trinity College Dublin  
**Term:** Spring 2026  

---

## Project Overview

This repository contains my final replication project for **Applied Statistical Analysis II**. The project replicates and extends Rossiter and Harden’s study, *The Development of an Issue Public: Evidence from The Eras Tour*.

The original paper examines how a politically meaningful **issue public** can emerge from a group initially connected by shared interest rather than prior political engagement. Using the Taylor Swift Eras Tour ticketing controversy as a case, the paper argues that a concrete and personalized experience of unfairness can activate issue-specific political attitudes and targeted political behavior among members of a potential issue public.

This project has two parts:

1. **Exact replication** of the core manuscript tables using the authors’ original replication code  
2. **Extension** testing whether the main findings are robust to the use of survey weights  

In this submission, the exact replication is centered on the manuscript’s core inferential tables. Figure 4 is discussed in the report and presentation as an original-study design figure used to explain the identification strategy, rather than as a newly reproduced figure output.

This submission is intentionally limited to the files required to reproduce the analyses discussed in my final project.

---

## Repository Scope

This repository contains the files required to reproduce the exact replication and extension reported in my final project.

It is therefore a project-specific subset of the original replication archive rather than a full re-upload of all original materials. The included files are sufficient to reproduce the analyses, tables, and extension outputs discussed in my report and presentation.

---

## Included Files

### Code

- `code/1_balance.R`
- `code/2_weighting.R`
- `code/3_estimation.R`
- `code/3b_make_table3.R`
- `code/10_extension_weighted_vs_unweighted.R`

### Data

- `data/processed_data/weighting.RData`

### Outputs

Recommended output files included in the repository:

- `tables/table1.txt`
- `tables/table2.txt`
- `tables/table3.txt`
- `tables/extension_weighted_vs_unweighted.csv`
- `graphs/extension_weighted_vs_unweighted_plot.png`

### Project documents

- `Replication presentation.pdf`
- `Original_text_and_summary_report.pdf`

---

## How to Reproduce

Set the working directory to the top level of this repository.

For the exact replication used in my report and presentation:

1. Run `code/3_estimation.R`
2. Run `code/3b_make_table3.R`

These scripts reproduce the core output files:

- `tables/table1.txt`
- `tables/table2.txt`
- `tables/table3.txt`

For the extension:

3. Run `code/10_extension_weighted_vs_unweighted.R`

This script produces:

- `tables/extension_weighted_vs_unweighted.csv`
- `graphs/extension_weighted_vs_unweighted_plot.png`

Note: `data/processed_data/weighting.RData` is already included, so it is not necessary to rerun the full preprocessing pipeline for this submission.

---

## Note on Software Compatibility

The supplementary script `code/3b_make_table3.R` is included to reproduce the placebo table output under the current package environment. This is necessary because the original placebo-table extraction depends on output formatting that is not fully stable across software versions.

---

## Repository Structure

```text
.
├── README.md
├── code/
│   ├── 1_balance.R
│   ├── 2_weighting.R
│   ├── 3_estimation.R
│   ├── 3b_make_table3.R
│   └── 10_extension_weighted_vs_unweighted.R
├── data/
│   └── processed_data/
│       └── weighting.RData
├── tables/
│   ├── table1.txt
│   ├── table2.txt
│   ├── table3.txt
│   └── extension_weighted_vs_unweighted.csv
├── graphs/
│   └── extension_weighted_vs_unweighted_plot.png
├── presentation/
│   └── Replication_presentation_QiLiu.pdf
└── Original_text_and_summary_report/
    └── final_written_report.pdf
    └──  Original_text.pdf
