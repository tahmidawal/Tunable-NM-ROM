# Nonlinear-decoder result audit

Generated from pulled artifacts and checksum manifests. Excluded cells are retained for diagnosis but do not support reported timing claims.

Overall accepted-result audit: **PASS**

## Cell integrity

| cell | job | device | disposition | checksums | launch provenance | GPU backend | f64/highest | complete | failure marker |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| nda_b128l5f16_r1 | 2652288 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_b160l4f16_r1 | 2652285 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_b160l4f31_r1 | 2652280 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bbench_g160_r12 | 2653828 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_be2e_g160_r12_failed | 2653868 | NVIDIA H200 | excluded: driver lost compact decoder metadata and aborted | True | True | True | True | False | True |
| nda_be2e_g160_r14 | 2656389 | NVIDIA H200 | excluded: raw timing repetitions were not persisted | True | True | True | True | True | False |
| nda_be2e_g160m592tau_r34 | 2664052 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_be2e_g160m640_r21 | 2662249 | NVIDIA A100 80GB PCIe | excluded: raw timing repetitions were not persisted | True | True | True | True | True | False |
| nda_be2e_g160m640_r24 | 2662656 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_beq128_s0_r16 | 2660747 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_beq128_s1_r16 | 2660771 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_beq128_s2_r16 | 2660790 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_beq128g4_s0_r20 | 2662191 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_beq128m592_s0_r32 | 2663710 | NVIDIA A100-SXM4-40GB | accepted | True | True | True | True | True | False |
| nda_beq128m592_s1_r32 | 2663713 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_beq128m592_s2_r32 | 2663721 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_beq128ref_s0_r25 | 2662856 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_beq128ref_s1_r25 | 2662859 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_beq128ref_s2_r25 | 2662860 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg144l4g2f31_s0_r29 | 2663383 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_bg159l4g3f31_s0_r22 | 2662575 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg159l4g3f31_s1_r22 | 2662581 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bg159l4g3f31_s2_r22 | 2662582 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bg160l4g1f31_r10 | 2653675 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg160l4g2f31_r2 | 2652520 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_bg160l4g2f31_s1_r11 | 2653792 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg160l4g2f31_s2_r11 | 2653799 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg160l4g4f31_r15 | 2658399 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bg160l4g4f31_s1_r18 | 2661924 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_bg160l4g4f31_s2_r18 | 2661927 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bg160l4g8f31_r19 | 2661961 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_bg192l4g2f31_r2 | 2652524 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_bobj160_r9 | 2653635 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_bobj160g4_r17 | 2661792 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_btrust160_r13 | 2653929 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_p128l4_r1 | 2652279 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_p96l4_r1 | 2652275 | NVIDIA L40S | accepted | True | True | True | True | True | False |
| nda_pbench_g2_r3 | 2652805 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_pbench_g98_r8 | 2653485 | NVIDIA H200 | excluded: exact kernels were warmed before the post-fit GPU burn | True | True | True | True | True | False |
| nda_pbench_g98b_r8 | 2653605 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pe2e_g98_r11 | 2653761 | NVIDIA H200 | excluded: raw timing repetitions were not persisted | True | True | True | True | True | False |
| nda_pe2e_g98_r23 | 2662651 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_pe2e_g98m448_r33 | 2663827 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_pe2e_g98m448tau_r35 | 2664042 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_peq104m416_s0_r30 | 2663614 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_peq104m416_s1_r30 | 2663620 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_peq104m416_s2_r30 | 2663621 | NVIDIA A100-SXM4-40GB | accepted | True | True | True | True | True | False |
| nda_peq108m432_s0_r31 | 2663700 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_peq108m432_s1_r31 | 2663702 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_peq108m432_s2_r31 | 2663708 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_peq112m448_s0_r28 | 2663252 | NVIDIA A100-SXM4-40GB | accepted | True | True | True | True | True | False |
| nda_peq112m448_s1_r28 | 2663257 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_peq112m448_s2_r28 | 2663259 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_peq96_s0_r26 | 2662896 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_peq96_s1_r26 | 2662903 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_peq96_s2_r26 | 2662904 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_peq96m480_s0_r27 | 2663233 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_peq96m480_s1_r27 | 2663236 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |
| nda_peq96m480_s2_r27 | 2663239 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_pg100l4g2_r5 | 2653237 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_pg100l4g2_s1_r7 | 2653403 | NVIDIA A100 80GB PCIe | accepted | True | True | True | True | True | False |
| nda_pg100l4g2_s2_r7 | 2653407 | NVIDIA L40S | accepted | True | True | True | True | True | False |
| nda_pg104l4g2_r4 | 2653134 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pg112l3g2_r3 | 2652975 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pg112l4g2_r3 | 2652970 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pg128l4g2_r2 | 2652509 | NVIDIA L40S | accepted | True | True | True | True | True | False |
| nda_pg128l4g4_r2 | 2652516 | NVIDIA A100-PCIE-40GB | accepted | True | True | True | True | True | False |
| nda_pg96l4g2_r3 | 2652974 | NVIDIA L40S | accepted | True | True | True | True | True | False |
| nda_pg98l4g2_r6 | 2653310 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pg98l4g2_s1_r7 | 2653395 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pg98l4g2_s2_r7 | 2653397 | NVIDIA H200 | accepted | True | True | True | True | True | False |
| nda_pvalid_g98_r8 | 2653487 | NVIDIA H100 PCIe | accepted | True | True | True | True | True | False |

## Accepted timing arrays

| cell | present | repetition audit | details |
|---|---:|---:|---|
| nda_pbench_g98b_r8 | True | True | raw arrays and medians agree |
| nda_bbench_g160_r12 | True | True | raw arrays and medians agree |
| nda_be2e_g160m592tau_r34 | True | True | raw arrays and medians agree |
| nda_be2e_g160m640_r24 | True | True | raw arrays and medians agree |
| nda_pe2e_g98_r23 | True | True | raw arrays and medians agree |
| nda_pe2e_g98m448_r33 | True | True | raw arrays and medians agree |
| nda_pe2e_g98m448tau_r35 | True | True | raw arrays and medians agree |
