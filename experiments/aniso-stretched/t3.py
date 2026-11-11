import numpy as np, time, sys
import aniso3d as a3, solve3d as s3
ev = a3.Evaluator3(128); P = a3.sample_params("dev", 16)
i = int(sys.argv[1]); mu = P[i]; print("mu", mu, flush=True)
fields = {}
for p, N in ((4, 32), (4, 24), (4, 16), (2, 32), (1, 64)):
    S = s3.Solver(N, p); t = time.time(); x, k, res = S.solve(mu, tol=1e-10, prec="fd"); x = np.asarray(x)
    fields[(p, N)] = ev.field(S.sp, x)
    print(p, N, S.n ** 3, "it", k, "res %.1e" % res, "t %.1fs" % (time.time() - t), {kk: "%.5f" % v for kk, v in ev.errors(fields[(p, N)], fields[(4, 32)]).items()}, flush=True)
