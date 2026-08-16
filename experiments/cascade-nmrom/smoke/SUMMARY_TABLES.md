# Cascade NM-ROM summary tables (machine-generated)


## cn_burgers_enc_K4_X0_report.json  (complete=True)


### stages
| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |
|---|---|---|---|---|---|---|
| 0 | 2.40e-01 | 6 | 7.995e-01 | 7.576e-01 | 9.387e-01 | 9.480e-01 |
| 1 | 1.92e-01 | 6 | 8.010e-01 | 7.562e-01 | 1.068e+00 | 9.829e-01 |

### residual probes (before next stage)
| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |
|---|---|---|---|---|---|---|
| -1 | 3.7 | 1.14e-01 | 1.29e-03 | 0.23 | 0.24 | False |
| 0 | 3.7 | 1.16e-01 | 2.44e-03 | -0.02 | -0.01 | True |
| 1 | 3.7 | 1.17e-01 | 2.50e-03 | -0.02 | -0.01 | True |

### ROM rollout (held-out, z0 = E(u0))
| stages | mean all-t | final-t | t0 encoded | inferred-latent all-t | acc/step |
|---|---|---|---|---|---|
| 1 | 1.045e+00 | 1.063e+00 | 1.028e+00 | 9.782e-01 | 3 |
| 2 | 1.128e+00 | 1.141e+00 | 1.115e+00 | 9.866e-01 | 3 |

## cn_poisson_enc_K4_X0_S0_report.json  (complete=True)


### stages
| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |
|---|---|---|---|---|---|---|
| 0 | 2.88e-03 | 6 | 6.320e-02 | 9.615e-02 | 1.172e+00 | 1.347e+00 |
| 1 | 1.82e-04 | 6 | 3.870e-02 | 5.418e-02 | 1.201e+00 | 1.320e+00 |

### residual probes (before next stage)
| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |
|---|---|---|---|---|---|---|
| -1 | 2.9 | 1.10e-01 | 0.00e+00 | 0.81 | 0.70 | False |
| 0 | 11.5 | 5.35e-01 | 0.00e+00 | 0.28 | 0.13 | True |
| 1 | 13.7 | 5.76e-01 | 0.00e+00 | -0.00 | -0.01 | True |

### held-out finite-budget inferred latents (data misfit LM)
| stages | init encoded | init mean | best |
|---|---|---|---|
| 1 | 2.392e-01 | 2.392e-01 | 2.392e-01 |
| 2 | 2.258e-01 | 2.258e-01 | 2.258e-01 |

### ROM (held-out, init = E(f))
| objective | colloc | stages | ROM mean | med | max | encoded plug-in | inferred | r_lm | r_enc | acc/rej |
|---|---|---|---|---|---|---|---|---|---|---|
| fd | full | 1 | 4.561e-01 | 4.561e-01 | 6.315e-01 | 8.435e-01 | 2.392e-01 | 7.97e-01 | 1.12e+00 | 5/0 |
| fd | full | 2 | 4.244e-01 | 4.244e-01 | 6.252e-01 | 8.041e-01 | 2.258e-01 | 7.83e-01 | 1.10e+00 | 5/0 |
| fd | m64 | 1 | 4.721e-01 | 4.721e-01 | 7.806e-01 | 8.435e-01 | 2.392e-01 | 3.23e-01 | 6.26e-01 | 5/0 |
| fd | m64 | 2 | 4.421e-01 | 4.421e-01 | 7.336e-01 | 8.041e-01 | 2.258e-01 | 3.30e-01 | 6.60e-01 | 5/0 |
| hinv | full | 1 | 2.396e-01 | 2.396e-01 | 3.756e-01 | 8.435e-01 | 2.392e-01 | 1.82e-03 | 7.60e-03 | 5/0 |
| hinv | full | 2 | 2.263e-01 | 2.263e-01 | 3.593e-01 | 8.041e-01 | 2.258e-01 | 1.68e-03 | 7.44e-03 | 5/0 |
| hinvK | full | 1 | 2.457e-01 | 2.457e-01 | 3.784e-01 | 8.435e-01 | 2.392e-01 | 1.76e-03 | 4.63e-03 | 4/0 |
| hinvK | full | 2 | 2.334e-01 | 2.334e-01 | 3.636e-01 | 8.041e-01 | 2.258e-01 | 1.62e-03 | 4.61e-03 | 5/0 |

## cn_poisson_enc_K4_X2_S0.1_report.json  (complete=True)


### stages
| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |
|---|---|---|---|---|---|---|
| 0 | 2.88e-03 | 6 | 2.259e-01 | 3.050e-01 | 8.410e-01 | 1.387e+00 |

### residual probes (before next stage)
| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |
|---|---|---|---|---|---|---|
| -1 | 2.9 | 1.10e-01 | 0.00e+00 | 0.78 | 0.68 | False |
| 0 | 7.9 | 3.72e-01 | 0.00e+00 | 0.31 | 0.13 | True |

### held-out finite-budget inferred latents (data misfit LM)
| stages | init encoded | init mean | best |
|---|---|---|---|
| 1 | 3.688e-01 | 3.703e-01 | 3.688e-01 |

### ROM (held-out, init = E(f))
| objective | colloc | stages | ROM mean | med | max | encoded plug-in | inferred | r_lm | r_enc | acc/rej |
|---|---|---|---|---|---|---|---|---|---|---|
| fd | full | 1 | 5.811e-01 | 5.811e-01 | 7.107e-01 | 8.019e-01 | 3.688e-01 | 9.23e-01 | 1.22e+00 | 3/0 |
| fd | m64 | 1 | 6.540e-01 | 6.540e-01 | 9.840e-01 | 8.019e-01 | 3.688e-01 | 4.02e-01 | 6.12e-01 | 3/0 |

## cn_poisson_enc_K4_X2_S0_report.json  (complete=True)


### stages
| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |
|---|---|---|---|---|---|---|
| 0 | 2.88e-03 | 6 | 2.078e-01 | 2.958e-01 | 1.153e+00 | 1.434e+00 |
| 1 | 5.97e-04 | 6 | 1.138e-01 | 1.646e-01 | 1.446e+00 | 1.427e+00 |
| 2 | 3.27e-04 | 6 | 8.444e-02 | 1.233e-01 | 1.452e+00 | 1.424e+00 |

### residual probes (before next stage)
| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |
|---|---|---|---|---|---|---|
| -1 | 2.9 | 1.10e-01 | 0.00e+00 | 0.80 | 0.67 | False |
| 0 | 6.8 | 3.81e-01 | 0.00e+00 | 0.27 | 0.10 | True |
| 1 | 10.6 | 5.07e-01 | 0.00e+00 | 0.21 | 0.05 | True |
| 2 | 10.1 | 5.31e-01 | 0.00e+00 | 0.05 | -0.02 | True |

### held-out finite-budget inferred latents (data misfit LM)
| stages | init encoded | init mean | best |
|---|---|---|---|
| 1 | 5.453e-01 | 7.073e-01 | 5.040e-01 |
| 2 | 2.887e-01 | 2.890e-01 | 2.887e-01 |
| 3 | 2.778e-01 | 2.782e-01 | 2.778e-01 |

### ROM (held-out, init = E(f))
| objective | colloc | stages | ROM mean | med | max | encoded plug-in | inferred | r_lm | r_enc | acc/rej |
|---|---|---|---|---|---|---|---|---|---|---|
| fd | full | 1 | 5.922e-01 | 5.922e-01 | 8.269e-01 | 9.816e-01 | 5.453e-01 | 8.96e-01 | 1.16e+00 | 2/0 |
| fd | full | 2 | 4.559e-01 | 4.559e-01 | 6.546e-01 | 9.185e-01 | 2.887e-01 | 8.66e-01 | 1.17e+00 | 3/0 |
| fd | full | 3 | 4.191e-01 | 4.191e-01 | 5.926e-01 | 9.228e-01 | 2.778e-01 | 8.55e-01 | 1.15e+00 | 3/0 |
| fd | m64 | 1 | 6.395e-01 | 6.395e-01 | 8.495e-01 | 9.816e-01 | 5.453e-01 | 3.97e-01 | 5.99e-01 | 3/0 |
| fd | m64 | 2 | 4.840e-01 | 4.840e-01 | 7.510e-01 | 9.185e-01 | 2.887e-01 | 3.71e-01 | 6.20e-01 | 3/0 |
| fd | m64 | 3 | 4.716e-01 | 4.716e-01 | 7.392e-01 | 9.228e-01 | 2.778e-01 | 3.61e-01 | 6.09e-01 | 3/0 |
| hinvK | full | 1 | 5.887e-01 | 5.887e-01 | 8.767e-01 | 9.816e-01 | 5.453e-01 | 3.88e-03 | 6.34e-03 | 2/0 |
| hinvK | full | 2 | 3.002e-01 | 3.002e-01 | 4.187e-01 | 9.185e-01 | 2.887e-01 | 2.32e-03 | 5.28e-03 | 3/0 |
| hinvK | full | 3 | 2.849e-01 | 2.849e-01 | 4.057e-01 | 9.228e-01 | 2.778e-01 | 2.23e-03 | 5.26e-03 | 3/0 |

## cn_poisson_truez_K4_X0_S0_report.json  (complete=True)


### stages
| stage | eps_in | n_freq | train global | train mean | held-out (encoded) global | mean |
|---|---|---|---|---|---|---|
| 0 | 2.88e-03 | 6 | 2.611e-01 | 2.541e-01 | 3.386e-01 | 4.397e-01 |
| 1 | 7.51e-04 | 6 | 2.401e-01 | 2.150e-01 | 3.217e-01 | 4.747e-01 |

### residual probes (before next stage)
| after stage | eff_rank | pod8 err | pod32 err | nn1 corr | nn5 corr | stop? |
|---|---|---|---|---|---|---|
| -1 | 2.9 | 1.10e-01 | 0.00e+00 | 0.82 | 0.71 | False |
| 0 | 4.2 | 2.97e-01 | 0.00e+00 | 0.27 | 0.04 | True |
| 1 | 4.4 | 2.89e-01 | 0.00e+00 | 0.22 | -0.02 | True |

### held-out finite-budget inferred latents (data misfit LM)
| stages | init encoded | init mean | best |
|---|---|---|---|
| 1 | 3.092e-01 | 3.159e-01 | 3.092e-01 |
| 2 | 2.760e-01 | 2.919e-01 | 2.760e-01 |

### ROM (held-out, init = E(f))
| objective | colloc | stages | ROM mean | med | max | encoded plug-in | inferred | r_lm | r_enc | acc/rej |
|---|---|---|---|---|---|---|---|---|---|---|
| fd | full | 1 | 5.653e-01 | 5.653e-01 | 5.786e-01 | 4.028e-01 | 3.092e-01 | 9.11e-01 | 1.10e+00 | 2/1 |
| fd | full | 2 | 5.373e-01 | 5.373e-01 | 6.367e-01 | 3.902e-01 | 2.760e-01 | 8.66e-01 | 1.04e+00 | 2/0 |
| fd | m64 | 1 | 7.845e-01 | 7.845e-01 | 9.606e-01 | 4.028e-01 | 3.092e-01 | 4.23e-01 | 5.78e-01 | 2/0 |
| fd | m64 | 2 | 6.671e-01 | 6.671e-01 | 9.579e-01 | 3.902e-01 | 2.760e-01 | 3.76e-01 | 5.26e-01 | 3/0 |
