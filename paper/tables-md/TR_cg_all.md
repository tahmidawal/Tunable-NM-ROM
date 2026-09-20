| Problem | $N$ | Method | Error (\%) | GPU ms | CG error (\%) | CG ms | $S$ | CG setting | Job |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson | 256 | NM-ROM, $q=0$ | 3.1567 | 6.7031 | 0.1527 | 22.2898 | 3.33$\times$ | cg\_0.01 | \texttt{3780692} |
| Poisson | 256 | NM-ROM, $q=256$ | 0.9689 | 7.1213 | 0.1527 | 22.2898 | 3.13$\times$ | cg\_0.01 | \texttt{3780692} |
| Poisson | 256 | Linear-bank ROM | 0.7459 | 1.6928 | 0.1527 | 22.2898 | 13.17$\times$ | cg\_0.01 | \texttt{3780692} |
| Poisson | 1024 | NM-ROM, $q=0$ | 3.1495 | 3.9406 | 0.0722 | 60.0170 | 15.23$\times$ | cg\_0.01 | \texttt{3783813} |
| Poisson | 1024 | NM-ROM, $q=256$ | 0.9648 | 4.3188 | 0.0722 | 60.0170 | 13.90$\times$ | cg\_0.01 | \texttt{3783813} |
| Poisson | 1024 | Linear-bank ROM | 0.7421 | 1.8300 | 0.0722 | 60.0170 | 32.80$\times$ | cg\_0.01 | \texttt{3783813} |
| Wave | 64 | NM-ROM, $q=0$ | 6.3109 | 184.5351 | 2.7812 | 41.8936 | 0.23$\times$ | cgdt\_0.005\_tol\_0.01 | \texttt{3780447} |
| Wave | 64 | NM-ROM, $q=32$ | 2.6815 | 2500.0601 | 1.5967 | 69.3737 | 0.03$\times$ | cgdt\_0.005\_tol\_1e-06 | \texttt{3780447} |
| Wave | 64 | Linear-bank ROM | 2.6913 | 1.1685 | 1.5967 | 69.3737 | 59.37$\times$ | cgdt\_0.005\_tol\_1e-06 | \texttt{3780447} |
| Wave | 256 | NM-ROM, $q=0$ | 6.1035 | 183.5155 | 0.4008 | 142.5111 | 0.78$\times$ | cg\_1e-06 | \texttt{3783805} |
| Wave | 256 | NM-ROM, $q=32$ | 2.8017 | 2520.1766 | 0.4008 | 142.5111 | 0.06$\times$ | cg\_1e-06 | \texttt{3783805} |
| Wave | 256 | Linear-bank ROM | 2.8115 | 1.4277 | 0.4008 | 142.5111 | 99.82$\times$ | cg\_1e-06 | \texttt{3783805} |
| Wave | 1024 | NM-ROM, $q=0$ | 6.1066 | 192.4624 | 5.1413 | 583.0259 | 3.03$\times$ | cgdt\_0.005\_tol\_0.01 | \texttt{3780450} |
| Wave | 1024 | NM-ROM, $q=32$ | 2.8094 | 2529.4450 | 0.4010 | 1043.3681 | 0.41$\times$ | cg\_1e-06 | \texttt{3780450} |
| Wave | 1024 | Linear-bank ROM | 2.8192 | 4.5624 | 0.4010 | 1043.3681 | 228.69$\times$ | cg\_1e-06 | \texttt{3780450} |
| Heat | 64 | NM-ROM | 4.5593 | 11.2268 | 1.5860 | 2.5038 | 0.22$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
| Heat | 64 | Linear-bank ROM | 1.6758 | 0.1083 | 1.5860 | 2.5038 | 23.12$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
| Heat | 256 | NM-ROM | 4.5557 | 11.3414 | 0.9255 | 6.9389 | 0.61$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
| Heat | 256 | Linear-bank ROM | 1.6758 | 0.1382 | 0.9255 | 6.9389 | 50.23$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
| Heat | 1024 | NM-ROM | 4.5555 | 12.2062 | 0.7707 | 59.1780 | 4.85$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
| Heat | 1024 | Linear-bank ROM | 1.6758 | 0.8381 | 0.7707 | 59.1780 | 70.61$\times$ | fom\_cg\_cn\_tol1e2 | \texttt{3529772} |
