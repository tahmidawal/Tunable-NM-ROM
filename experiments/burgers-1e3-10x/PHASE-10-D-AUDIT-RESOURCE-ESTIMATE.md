# Phase-10-D audit-only portability resource estimate

The single audit-only cell requests one H200, 8 CPUs, 96 GiB host memory, and
16 hours. It performs zero optimizer or scientific updates. It checks the
immutable r2 trace and repeats train-data regeneration plus initial/terminal
full-grid Cox/K3 evaluation. The failed r2 job completed those finalization
checks in minutes after its 93-minute diagnostic; the 16-hour cap preserves a
large infrastructure margin while matching the source-cell device class.
