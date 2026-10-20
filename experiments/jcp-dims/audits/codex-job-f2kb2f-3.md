**f2kA: SUBMIT when capacity and budget allow.**  
**b2f: SUBMIT when capacity and budget allow.**

A9.3 blockers are resolved:

- The decoder checkpoint loaded by `Q.Model` is included in `NUMERIC_FILES`. In-memory hash mutation changes the attempt signature.
- `load_all()` aborts on mismatched configurations, including unique arms and duplicates already encountered. Matching configurations pass CPU fixtures.
- Both staged runs pass **176/176 manifest hashes**. Their only source difference is `analyze2d.py`; production inputs and code remain unchanged.

**Neither can submit now:** live queue is **3/3 lane, 8/8 account**. Both remote directories remain absent.

| Reservation scenario | Total against 55 GPU-h |
|---|---:|
| f2kA alone | 57.3879 — blocked |
| b2f alone | 52.6979 — fits |
| Both | 58.4279 — blocked |

Before submission, reconcile completed jobs and require **spent + live reservations + requested limits + remaining pending projections ≤ 55 GPU-h**, lane count **<3**, and account count **<8**, using the shared submission lock.

No files changed, jobs submitted, or LAB-LOG written.