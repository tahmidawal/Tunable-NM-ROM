Smoke run (job 7174901, H200, smallest size of each family), kept for the record; not used in any table.
The AmgX files in this directory are INVALID for repeats 2-6: `AMGX_solver_solve_with_0_initial_guess`
assumes the solution vector is zero without zeroing it, and the vector was not reset between repeats, so
the externally computed residual grows by one per repeat (PCG, BiCGStab) or the solve returns after one
iteration (FGMRES).  Fixed before the production runs (see PROTOCOL.md, amendment 1).
