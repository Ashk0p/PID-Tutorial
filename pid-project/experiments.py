"""Recreates every figure and number used in the PID report.
Run:  python experiments.py      (figures -> ./figures, numbers -> results.json)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pid_lib import PID, SpringMassDamper, ThirdOrderLag, simulate, metrics

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
M, C, K = 1.0, 10.0, 20.0            # kg, N*s/m, N/m
FINAL = dict(kp=600.0, ki=200.0, kd=30.0)   # chosen in section 6
RES, CAPTIONS = {}, {}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})

def new(title, xl="Time (s)", yl="Position (m)", size=(6.4, 3.8)):
    fig, ax = plt.subplots(figsize=size); ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
    return fig, ax

def save(fig, name, caption):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); CAPTIONS[name] = caption

def run(kp, ki=0, kd=0, T=3.0, **kw):
    r = simulate(SpringMassDamper(M, C, K), PID(kp, ki, kd, **kw.pop("pid", {})), kw.pop("sp", 1.0), T=T, **kw)
    return r, metrics(r["t"], r["y"])

def tgt(ax, v=1.0): ax.axhline(v, color="k", ls="--", lw=1, label="Target 1 m")

# ---- Fig 1: loop diagram -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 2.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
def box(x, w, txt, col):
    ax.add_patch(FancyBboxPatch((x, 1.1), w, 1, boxstyle="round,pad=0.05", fc=col, ec="k")); ax.text(x + w/2, 1.6, txt, ha="center", va="center", fontsize=9)
box(0.1, 1.4, "Target\n(setpoint)", "#fde7cf"); box(3.2, 1.8, "Controller\n(PID)", "#cfe3fd"); box(5.8, 1.6, "Machine\n(plant)", "#d5f0d5"); box(8.3, 1.4, "Sensor", "#eee")
for x0, x1, y in [(1.5, 3.2, 1.6), (5.0, 5.8, 1.6), (7.4, 8.3, 1.6)]: ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="->"))
ax.text(2.35, 1.8, "error\n= target − actual", ha="center", fontsize=7); ax.text(5.4, 1.8, "force u", ha="center", fontsize=7)
ax.plot([9, 9, 0.8, 0.8], [1.1, 0.35, 0.35, 1.05], "k-"); ax.annotate("", (0.8, 1.1), (0.8, 0.4), arrowprops=dict(arrowstyle="->"))
ax.text(5, 0.1, "measured value fed back (the 'loop')", ha="center", fontsize=8); ax.set_title("Closed-loop feedback: target → controller → machine → sensor → back")
save(fig, "fig01_loop", "The feedback loop: the controller acts on the gap between target and measurement.")

# ---- Fig 2: open loop, why spring needs constant push ----------------------------
fig, ax = new("Open loop: constant push of 20 N holds the block at 1 m")
for u0 in [10, 20, 30]:
    pl = SpringMassDamper(); n = 3000; y = []
    for _ in range(n): y.append(pl.y); pl.step(u0, .001)
    ax.plot(np.arange(n)*.001, y, label=f"u = {u0} N (rests at {u0/K:.2f} m)")
tgt(ax); ax.legend()
save(fig, "fig02_openloop", "Final position = u/k, so holding 1 m needs u = k*1 = 20 N forever (the spring pulls back).")
RES["hold_force_N"] = K * 1.0
RES["poles"] = [float(p) for p in np.roots([M, C, K])]

# ---- Fig 3: four measures marked -------------------------------------------------
r, m = run(600, 200, 0, T=3)
t, y = r["t"], r["y"]; fig, ax = new("The four measures on one step response (P=600, I=200)")
ax.plot(t, y, label="Position"); tgt(ax)
t10, t90 = t[np.argmax(y >= .1)], t[np.argmax(y >= .9)]
ax.axvspan(t10, t90, color="C1", alpha=.25, label=f"1 Rise time {m['rise']*1000:.0f} ms (10→90 %)")
ax.plot(t[np.argmax(y)], y.max(), "rv", label=f"2 Overshoot {m['overshoot']:.1f} %")
ax.axhspan(.98, 1.02, color="g", alpha=.12, label="±2 % band")
ts = m["settle"]
if ts == ts: ax.axvline(ts, color="g", ls=":", label=f"3 Settling time {ts:.2f} s")
ax.annotate(f"4 Steady-state error ≈ {m['sse']*1000:.1f} mm", (2.2, .985), (1.2, .8), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(0, 2.5); ax.legend(fontsize=7, loc="lower right")
save(fig, "fig03_measures", "Rise time, overshoot, settling time and steady-state error marked on one response.")
RES["fig3_metrics"] = m

# ---- Fig 4/5/6: P alone, + I, + D --------------------------------------------------
fig, ax = new("P only: higher Kp → faster, smaller offset, never zero")
RES["P_only"] = {}
for kp in [5, 20, 100, 500]:
    r, m = run(kp); ax.plot(r["t"], r["y"], label=f"Kp={kp}")
    RES["P_only"][kp] = dict(m, sse_theory=K/(K+kp), y_final_theory=kp/(K+kp))
tgt(ax); ax.legend(); save(fig, "fig04_p_only", "P-only stops at Kp/(k+Kp): 0.20, 0.50, 0.83, 0.96 m for Kp=5,20,100,500.")

fig, ax = new("Check: simulated vs calculated steady-state error (P only)", "Kp (N/m)", "Steady-state error (m)")
kps = np.logspace(0, 3, 40); sim = [run(k, T=4)[1]["sse"] for k in kps]
ax.loglog(kps, K/(K+kps), "k-", label="Calculation: k/(k+Kp)"); ax.loglog(kps, sim, "ro", ms=4, label="Simulation"); ax.legend()
save(fig, "fig05_sse_check", "Simulation dots sit on the formula line: the calculation matches the graphs.")

fig, ax = new("P + I (Kp=100): integral removes the offset"); RES["PI"] = {}
for ki in [0, 50, 200, 800]:
    r, m = run(100, ki, 0, T=4); ax.plot(r["t"], r["y"], label=f"Ki={ki}"); RES["PI"][ki] = m
tgt(ax); ax.legend(); save(fig, "fig06_pi", "Adding I drives the error to ~0 but larger Ki adds overshoot/oscillation.")

fig, ax = new("P + D (Kp=300): derivative damps the bounce"); RES["PD"] = {}
for kd in [0, 5, 10, 20]:
    r, m = run(300, 0, kd, T=2); ax.plot(r["t"], r["y"], label=f"Kd={kd}"); RES["PD"][kd] = m
tgt(ax); ax.legend(); save(fig, "fig07_pd", "Kd=0 overshoots ~32 %; Kd=20 gives 0 % overshoot (offset 0.0625 m remains).")
# damping calc for P+D: s^2 + (c+Kd)s + (k+Kp) -> zeta
RES["zeta"] = {kd: (C+kd)/(2*np.sqrt(M*(K+300))) for kd in [0, 5, 10, 20]}
RES["overshoot_theory_%"] = {kd: float(100*np.exp(-np.pi*z/np.sqrt(1-z**2))) if z < 1 else 0.0 for kd, z in RES["zeta"].items()}

# ---- Fig 8: compare P, PI, PD, PID ---------------------------------------------------
fig, ax = new("P vs PI vs PD vs PID (Kp=600, Ki=200, Kd=30)", size=(6.8, 4)); RES["compare"] = {}
for name, g in [("P", (600, 0, 0)), ("PI", (600, 200, 0)), ("PD", (600, 0, 30)), ("PID", (600, 200, 30))]:
    r, m = run(*g, T=3); ax.plot(r["t"], r["y"], label=name); RES["compare"][name] = m
tgt(ax); ax.set_xlim(0, 1.5); ax.legend(); save(fig, "fig08_compare", "PID is fast, has ~0 overshoot and zero final error; each other combo lacks something.")

# ---- Section 7 problems and fixes -----------------------------------------------------
LIM = 30.0
fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True); RES["windup"] = {}
for name, aw, col in [("No anti-windup", "none", "C3"), ("Back-calculation", "backcalc", "C2")]:
    r, m = run(100, 800, 20, T=4, pid=dict(u_min=-LIM, u_max=LIM, antiwindup=aw, d_on="measurement", kb=20.0))
    axs[0].plot(r["t"], r["y"], col, label=name); axs[1].plot(r["t"], r["I"], col, label=name); RES["windup"][name] = m
axs[0].axhline(1, color="k", ls="--"); axs[0].set_title("Windup with a 30 N force limit (Kp=100, Ki=800, Kd=20)"); axs[0].set_ylabel("Position (m)"); axs[0].legend()
axs[1].set_ylabel("Integral term (N)"); axs[1].set_xlabel("Time (s)")
save(fig, "fig09_windup", "Without anti-windup the I term charges up while saturated and causes overshoot; back-calculation removes it.")

fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True); RES["noise"] = {}
for name, tau, col in [("Raw D", 0.0, "C3"), ("Filtered D (τ=20 ms)", 0.02, "C2")]:
    r, m = run(600, 200, 30, T=2, noise=0.001, pid=dict(d_tau=tau, d_on="measurement"))
    axs[0].plot(r["t"], r["u"], col, lw=.6, label=name); axs[1].plot(r["t"], r["y"], col, label=name)
    RES["noise"][name] = dict(m, u_std=float(np.std(r["u"][1000:])))
axs[0].set_title("Sensor noise ±1 mm: D amplifies it"); axs[0].set_ylabel("Control force (N)"); axs[0].legend(); axs[1].set_ylabel("Position (m)"); axs[1].set_xlabel("Time (s)")
save(fig, "fig10_noise", "Raw D makes the force jitter wildly; a low-pass filter on D cuts the jitter while keeping the response.")

fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True); RES["kick"] = {}
for name, don, col in [("D on error", "error", "C3"), ("D on measurement", "measurement", "C2")]:
    r, m = run(600, 200, 30, T=1.5, sp=lambda t: 0.5 if t < .5 else 1.0, pid=dict(d_on=don))
    axs[0].plot(r["t"], r["D"], col, label=name); axs[1].plot(r["t"], r["y"], col, label=name)
    RES["kick"][name] = float(np.max(np.abs(r["D"])))
axs[0].set_yscale("symlog"); axs[0].set_title("Derivative kick when target jumps 0.5→1 m at t=0.5 s"); axs[0].set_ylabel("D term (N, symlog)"); axs[0].legend()
axs[1].set_ylabel("Position (m)"); axs[1].set_xlabel("Time (s)")
save(fig, "fig11_kick", "Differentiating the error gives a one-step spike (Kd/dt); differentiating the measurement removes it.")

fig, ax = new("Slow controller updates (same gains)", size=(6.8, 4)); RES["slow"] = {}
for cdt in [0.001, 0.01, 0.03, 0.06, 0.1]:
    r, m = run(600, 200, 30, T=2, pid=dict(dt=cdt, d_on="measurement"))
    ax.plot(r["t"], np.clip(r["y"], -3, 3), label=f"dt={cdt*1000:g} ms"); RES["slow"][cdt] = m
tgt(ax); ax.set_ylim(-2, 3); ax.legend(fontsize=7); save(fig, "fig12_slow", "Slower updates add delay; beyond ~dt = 50 ms these gains oscillate and diverge.")

# ---- Section 8: manual tune + Ziegler-Nichols -----------------------------------------
fig, ax = new("Manual tuning, step by step", size=(6.8, 4)); RES["manual"] = {}
for name, g in [("1 P only (Kp=300)", (300, 0, 0)), ("2 + D (Kd=30)", (300, 0, 30)), ("3 raise Kp (600)", (600, 0, 30)), ("4 + I (Ki=200)", (600, 200, 30))]:
    r, m = run(*g, T=2); ax.plot(r["t"], r["y"], label=name); RES["manual"][name] = m
tgt(ax); ax.set_xlim(0, 1.2); ax.legend(fontsize=7); save(fig, "fig13_manual", "Typical hand-tuning order: P, then D to tame overshoot, then more P, then a little I.")

DT = 0.01
def zn_run(kp, T=120):
    rr = simulate(ThirdOrderLag(), PID(kp, dt=DT), 1.0, T=T, dt=DT); return rr
def growth(kp):
    y = zn_run(kp)["y"]; n = len(y); a = np.ptp(y[n//2:3*n//4]); b = np.ptp(y[3*n//4:]); return b / max(a, 1e-12)
lo, hi = 1.0, 20.0
for _ in range(40):
    mid = (lo + hi) / 2
    if growth(mid) > 1.0: hi = mid
    else: lo = mid
Ku = (lo + hi) / 2
y = zn_run(Ku)["y"]; tt = np.arange(len(y)) * DT
cross = tt[1:][(y[:-1] - y.mean() < 0) & (y[1:] - y.mean() >= 0)]; Pu = float(np.mean(np.diff(cross[-6:])))
RES["ZN"] = dict(Ku=Ku, Pu=Pu, Ku_theory=8.0, Pu_theory=2*np.pi/np.sqrt(3))
tab = {"P": (0.5*Ku, 0, 0), "PI": (0.45*Ku, 0.45*Ku/(Pu/1.2), 0), "PID (classic)": (0.6*Ku, 0.6*Ku/(Pu/2), 0.6*Ku*Pu/8)}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); axs[0].plot(tt[:3000], zn_run(Ku, T=120)["y"][:3000]); axs[0].set_title(f"P only at Ku={Ku:.2f}: steady swing, Pu={Pu:.2f} s")
axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Output (unitless)"); axs[0].axhline(1, color="k", ls="--")
RES["ZN_results"] = {}
tuned = dict(tab); tuned["PID improved (0.4Ku, Ti=Pu, filtered D)"] = (0.4*Ku, 0.4*Ku/Pu, 0.4*Ku*Pu/8)
for name, (kp, ki, kd) in tuned.items():
    pid = PID(kp, ki, kd, dt=DT, d_tau=0.05 if "improved" in name else 0.0, d_on="measurement")
    r = simulate(ThirdOrderLag(), pid, 1.0, T=30, dt=DT); m = metrics(r["t"], r["y"]); axs[1].plot(r["t"], r["y"], label=name); RES["ZN_results"][name] = dict(kp=kp, ki=ki, kd=kd, **m)
axs[1].axhline(1, color="k", ls="--", label="Target"); axs[1].set_title("Ziegler–Nichols tuned responses on 1/(s+1)³"); axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Output (unitless)"); axs[1].legend(fontsize=7)
for a in axs: a.grid(alpha=.3)
save(fig, "fig14_zn", "Left: P-only sustained swing gives Ku and Pu. Right: table gains are aggressive; reduced gains with D filtering overshoot less.")

# ---- Section 9: final output on the test system ---------------------------------------
r, m = run(**FINAL, T=3, pid=dict(u_min=-LIM*10, u_max=LIM*10, antiwindup="backcalc", d_on="measurement", d_tau=0.005))
RES["final_metrics"] = m
fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True)
axs[0].plot(r["t"], r["y"], label="Position"); axs[0].axhline(1, color="k", ls="--", label="Target"); axs[0].set_ylabel("Position (m)"); axs[0].set_title("Final PID on the test system"); axs[0].legend()
axs[1].plot(r["t"], r["P"], label="P"); axs[1].plot(r["t"], r["I"], label="I"); axs[1].plot(r["t"], r["D"], label="D"); axs[1].plot(r["t"], r["u"], "k", label="Total u"); axs[1].set_ylim(-300, 700)
axs[1].set_ylabel("Force (N)"); axs[1].set_xlabel("Time (s)"); axs[1].legend(ncol=4, fontsize=7)
save(fig, "fig15_final", "Final controller: response (top) and how P, I, D add up to the force (bottom; settles at 20 N).")

json.dump(RES, open("results.json", "w"), indent=1, default=float)
json.dump(CAPTIONS, open("figures/captions.json", "w"), indent=1)
print(json.dumps(RES, indent=1, default=float))
