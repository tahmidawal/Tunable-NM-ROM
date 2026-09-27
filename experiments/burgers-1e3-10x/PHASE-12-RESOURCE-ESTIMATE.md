# Phase-12 corrected-route resource estimate

The single prospective cell requests one H200, 8 CPUs, 96 GiB, and 16 hours.
The timed actual output is exactly 429,230,728 logical bytes in five f64 leaves;
the locked JAX 0.10.2 compiler output is 429,230,768 bytes and the complete
memory analysis must remain below the 20 GB preregistered eligibility cap.

The corrected structural preflight removes a 427,819,008-byte Cox control leaf
from every timed ROM invocation.  The separate identity executable is run once
per fixed case and never timed.  If structural speed or its lower bound misses,
the job stops before runtime projection and update 1.  If it passes, the
unchanged Phase-11 no-update projection charges the 18+54 conditional training
epochs, final-q globalization, possible 18 predictor epochs and selection,
compression, and audit; it proceeds only when 115% of projected remaining work
fits the actual Slurm time remaining.  This is the resource gate, not a promise
that training will consume the allocation.
