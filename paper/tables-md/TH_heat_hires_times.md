| Model | Mesh | Cohort (cases) | Stepping | ms acc. / fast | FOM ms | FOM (Table 1 rule) | Speedup | vs CN–CG 1e-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wide bank | 1024^2 | development (12) | CN | 31.2 / 36.9 | 66.8 | \Delta t=0.05, rtol 10^{-3} | 2.14 / 1.81× | 6.33 / 5.35× |
| Wide bank | 1024^2 | sealed (16) | CN | 27.3 / 27.8 | 43.5 | \Delta t=0.05, rtol 10^{-3} | 1.59 / 1.57× | 4.70 / 4.62× |
| Wide bank | 1024^2 | sealed (16) | batched fit | 3.1 / 3.4 | 43.5 | \Delta t=0.05, rtol 10^{-3} | 13.9 / 13.0× | 41.0 / 38.3× |
| Wide bank | 2048^2 | sealed (16) | CN | 28.0 / 28.4 | 272.5 | \Delta t=0.05, rtol 10^{-3} | 9.74 / 9.59× | 28.8 / 28.3× |
| Wide bank | 2048^2 | sealed (16) | batched fit | 4.5 / 4.8 | 272.5 | \Delta t=0.05, rtol 10^{-3} | 60.0 / 57.0× | 177 / 169× |
| Wide bank | 4096^2 | sealed (16) | CN | 31.7 / 32.2 | 1139.8 | \Delta t=0.05, rtol 10^{-2} | 36.0 / 35.4× | 149 / 146× |
| Wide bank | 4096^2 | sealed (16) | batched fit | 11.0 / 11.3 | 1139.8 | \Delta t=0.05, rtol 10^{-2} | 103 / 101× | 428 / 416× |
| Earlier (R=32) | 2048^2 | development (12) | CN | 15.9 / 8.9 | 173.1 | \Delta t=0.1, rtol 10^{-2} | 10.9 / 19.5× | 51.1 / 91.4× |
| Earlier (R=32) | 4096^2 | development (12) | CN | 18.0 / 10.8 | 1014.6 | \Delta t=0.1, rtol 10^{-2} | 56.5 / 93.7× | 267 / 442× |
| 3D, earlier bank | 128^3 | final (64) | CN | 35.1 / 16.5 | 7.0 | \Delta t=0.1, rtol 10^{-2} | 0.20 / 0.43× | 0.83 / 1.78× |
| 3D, earlier bank | 128^3 | final (64) | batched fit | 5.2 / 4.0 | 7.0 | \Delta t=0.1, rtol 10^{-2} | 1.36 / 1.77× | 5.67 / 7.40× |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | CN | 37.0 / 21.0 | 173.3 | \Delta t=0.05, rtol 10^{-4} | 4.69 / 8.27× | 8.96 / 15.8× |
| 3D, earlier bank | 256^3 | final, 1st 16 (16) | batched fit | 9.6 / 8.4 | 173.3 | \Delta t=0.05, rtol 10^{-4} | 18.1 / 20.5× | 34.7 / 39.3× |
