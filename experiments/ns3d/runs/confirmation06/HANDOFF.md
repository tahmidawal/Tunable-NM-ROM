# NS3D confirmation preflight failure

Job 4023444, source 9144209e87b76aa24fe492258d7ea6dc695b45f2, exited 1:0 after 26 seconds on pax049 after a valid GPU preflight. A `Path` indexing typo raised before training or evaluation. No scientific output or final case was generated. The original source and complete logs are retained in `collected/` and the round-trip verified `artifacts/confirmation06/` archive. The unchanged model bytes remain covered by the accepted coverage04 archive and the exact reuse SHA-256 manifest.

The path-join correction is verified by executing the actual source assertion against all retained checkpoint bytes, recorded in `checks/confirmation_reuse_preflight.json`. A fresh `confirmation06b` directory will carry the corrected committed source. The failed directory will be removed only after successor reuse copying and durable archive Git verification.
