# Manually-Created Data

This folder is used to hold manually created data that cannot be easily replicated.
You may keep data here and keep it under version control if it is small enough.
Keeping this data under version control can provide some peace of mind
that the data is not inadvertently modified.
Also, keep in mind that Git LFS is a good option if the data is large.

## suspension_hand_check.csv

Inferred dividend suspensions checked by hand against company filings. The
six firms are small banks drawn from a fixed random sample
(`src/diagnose_discrepancies.py`, added in PR D) of financial firms our rule flags in 2009.
For each: the date our rule expected a dividend, whether the firm in fact
stopped paying, how the suspension was disclosed, the evidence and its source.
`evidence_read` says whether the evidence was read word for word in the
source; all six rows have been.
