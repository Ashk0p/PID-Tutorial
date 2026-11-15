# PID Control Explained: Code, Report and Write-up

Hand-written PID controller (NumPy only, no control library), simulated on a 1 kg spring-damper block
(m = 1 kg, c = 10 N·s/m, k = 20 N/m, target 1 m) and on the Ziegler–Nichols plant 1/(s+1)^3.
Running one script recreates every graph and number in the PDF report.

## What's in this repo
| Path | Purpose |
|---|---|
| `pid_lib.py` | `PID` class (anti-windup, derivative-on-measurement, D filter, output limits), plants (`SpringMassDamper` with RK4, `ThirdOrderLag`), `simulate()`, `metrics()` |
| `experiments.py` | Generates all 15 figures into `figures/` and every quoted number into `results.json` |
| `build_report.py` | Rebuilds `PID_Report.pdf` from `results.json` + figures (optional) |
| `PID_Report.pdf` | The 10-section beginner's guide (10 pages) |
| `WRITEUP.md` | Approach, what worked, what didn't, what next |
| `figures/` | Generated PNGs + `captions.json` |

## Approach (short)
1. **Controller:** discrete PID, `u = Kp*e + Ki*sum(e*dt) + Kd*d/dt`, evaluated every controller step. Four real-world fixes are options in the class: output limit + back-calculation anti-windup, derivative on measurement, low-pass filter on D, no kick on the first call.
2. **Plant:** the block's equation `m x'' + c x' + k x = u` integrated with RK4 at 1 ms. The controller output is held constant between updates, so slow controllers can be simulated.
3. **Judging:** rise time (10→90 %), overshoot (%), settling time (±2 %), steady-state error (mean of last 5 % of run).
4. **Experiments:** P alone → +I → +D, all together, four failure cases, hand tuning and Ziegler–Nichols. Hand calculations (e.g. P-only error `k/(k+Kp)`, PD damping ratio) are checked against the simulation.

## Setup
Requires Python 3.9+.
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # numpy, matplotlib
```
Only for rebuilding the PDF: `pip install reportlab pillow`

## Run
```bash
python experiments.py     # ~1 min; writes figures/*.png, figures/captions.json, results.json
python build_report.py    # optional; writes PID_Report.pdf
```
Results are deterministic (fixed random seed for sensor noise), so you should get the same graphs and numbers.

## Quick use of the controller
```python
from pid_lib import PID, SpringMassDamper, simulate, metrics
pid = PID(600, 200, 30, dt=0.001, u_min=-300, u_max=300,
          antiwindup="backcalc", d_on="measurement", d_tau=0.005)
r = simulate(SpringMassDamper(1, 10, 20), pid, setpoint=1.0, T=3.0)
print(metrics(r["t"], r["y"]))
```

## Figure → report section
| Figure | Section | Figure | Section |
|---|---|---|---|
| fig01 loop diagram | 1 | fig09 windup | 7a |
| fig02 open loop | 3 | fig10 noise | 7b |
| fig03 four measures | 4 | fig11 derivative kick | 7c |
| fig04 P only, fig05 error check, fig06 +I, fig07 +D | 5 | fig12 slow updates | 7d |
| fig08 P/PI/PD/PID | 6 | fig13 manual tuning, fig14 Ziegler–Nichols | 8 |
| fig15 final controller | 9 | | |

## Conventions
Settling band ±2 %; steady-state error = target − mean of last 5 % of the window; "never" / NaN settling time = never entered the band within the run. All results are simulation only (no hardware).

## Interview "what if" parameters
Change these at the top of `experiments.py` (or in the call you care about): mass/damping/spring `M, C, K`, force limit `LIM`, final gains `FINAL`, noise `noise=` in the noise test, controller period `dt=`.
