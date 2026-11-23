# Third-party sources

`NativeDecision` implements the architecture and state-dictionary layout published in
`convaiinnovations/laya/rl_common.py`, revision
`e333cb7c823cd1787276228260f56d02fbf5180b` (Convai Innovations, Apache-2.0).
The task uses its binary choice scores; the auxiliary action head is retained for
checkpoint compatibility and is not trained.

`raw_source.clean_text` follows the text-cleaning operations in
JoakimEdin/medical-coding-reproducibility, revision
`70bda64dce532f2334f8b97dcd50c02f66dd232f` (Apache-2.0).
The source extraction does not compute label frequencies from held-out examples.

Benchmark candidate scopes and code descriptions are from JHLiu7/Dual-LAAT,
revision `d4f15e044afb0462077fe2134ddd633210bcd423`. The public reference
files are retained in `data/references/duallaat`; source cohort membership is
cross-checked against the Edin split release. The four-thousand-word limit
follows the preprocessing order in both repositories.

Dataset use and derived medical artifacts remain subject to the original
PhysioNet Credentialed Health Data License and Data Use Agreement.
