# Bounded native adapter interface control

This source candidate is unexecuted. It consumes the real unadmitted native01 producer
export under an exact-source coordinator release and runs only low-level adapter methods.
It neither calls the production TrainingFEAdapter loader/build/export nor fabricates a
training population, common lift or physical admission.

The new NativeInterface class explicitly reuses mapping, field evaluation, analytic
normal/gradient controls, component-map and metric assembly methods. It loads the exact
binary mesh, verifies ordered FE layout, reconstructs the saved velocity/pressure and
physical derivatives in dimensionless units, and compares every saved volume sample at
the existing 1e-10 tolerance. It assembles the scalar/vector metric blocks, checks their
component identities, then exercises the registered bounded scalar eigenproblems with
look-ahead, residual, Gram and orientation checks. Coefficients and all diagnostics remain
saved. These are interface controls, not reference or reduced-solution accuracy evidence.

Execution requires a new isolated ROM job directory, prescribed cluster interpreter,
GPU preflight, float64/highest, 1 GPU, at most 16 CPUs, 64 GiB and 20 minutes. Existing
adapter caps remain unchanged. The coarse producer has 415 elements, 4,889 mixed DOFs
and 20,335 volume points; the fixture evaluates one solution and bounded scalar eigenbanks,
not the large basis/training/quadrature-fit arrays. An 8 GiB working-set planning envelope
and 1 GiB result envelope fit the allocation and existing process/factor limits. These
are projections, not measured native memory. No run is authorized by this note.

A release record must bind schema `ns-obstacle-native-adapter-control-release-v1`, exact
`adapter_source_sha256`, `fixture_source_sha256`, real `producer_manifest_sha256`, and
`physical_admission=false`. All producer-declared artifact hashes are checked before
native imports. The coordinator reviews source and job packaging before execution.

## Glossary

FE: finite elements. Mapping: transforming integration points to physical curved elements.
Component map: the actual relation between scalar and vector FE coefficients. Metric:
assembled function-space inner product. Eigenbank: a fixed candidate set for later test
spaces. A look-ahead mode checks the cutoff gap; it is not proof of complete eigenvalue
counting. Admission: independent authorization of qualified physical reference data.
