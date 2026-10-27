# Operational cohort-normalisation restoration

Status: independently reviewed repair of a strict preflight failure; no solver,
quadrature or reference-eligibility threshold changes. The underlying process-level
floating-point path that changed a derived peak normaliser remains unknown.

The failed quadrature preflights remain in the ledger. The captured second failure's raw
table is preserved in `results/qpilot02/raw/output/regenerated_cohort.npz`, with the complete
collection checksum record. Independent review compared every approved unsealed E0 metadata
descriptor and confirmed that replacing only the affected derived normalisation value
restores the exact original complete cohort SHA, including all unused raw draw padding.
The actual failed snapshot is a small control of the repair.

Every subsequent job still regenerates the physical parameter table from its recorded seed.
It snapshots all raw arrays before inspecting or changing them. Approved normalisers come
only from the content-verified E0 ST metadata for the explicit validation cohort. A repair
is allowed only when all normalisers are positive finite float64 values and their exact
integer floating-point distances are at most one ULP. Only the derived normaliser vector
can change. The final full table fingerprint must equal the original frozen cohort
fingerprint exactly; otherwise execution stops before physics. This final requirement
rejects changes to any active or inactive raw draw.

Save the raw and restored tables, both fingerprints, changed case ids and exact differences.
Retain the existing independent E0 initial-field and norm checks. Changes beyond the bounded
case, altered seed fields, missing metadata or a failed final fingerprint remain failures.
This restores the originally approved physical-family inputs; it does not accept a newly
normalised cohort or select new cases.

## Glossary

- Normaliser: the stored peak used to scale an initial field to its prescribed amplitude.
- ULP: one adjacent representable floating-point value.
- Raw draw: a seeded physical parameter, including unused padded entries.
- Fingerprint: a SHA256 digest of the complete ordered input arrays.
- E0 ST: the approved space/time-refined reference and its recorded input descriptor.
