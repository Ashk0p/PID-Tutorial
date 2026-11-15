"""Hand-written PID controller, plants, simulator and step-response metrics.
Only NumPy is used (no control library)."""
import numpy as np


class PID:
    """Discrete PID with optional fixes for real-world problems.

    antiwindup : 'none' | 'backcalc'   (back-calculation when output saturates)
    d_on       : 'error' | 'measurement' (measurement avoids derivative kick)
    d_tau      : time constant (s) of first-order low-pass on D term (0 = off)
    """

    def __init__(self, kp, ki=0.0, kd=0.0, dt=0.001, u_min=-np.inf, u_max=np.inf,
                 antiwindup="none", d_on="error", d_tau=0.0, kb=None):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.u_min, self.u_max = u_min, u_max
        self.antiwindup, self.d_on, self.d_tau = antiwindup, d_on, d_tau
        self.kb = kb if kb is not None else (ki / kp if kp > 0 and ki > 0 else 0.0)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_e = None
        self.prev_y = None
        self.d_filt = 0.0
        self.last = (0.0, 0.0, 0.0)  # P, I, D contributions

    def update(self, setpoint, y):
        e = setpoint - y
        if self.prev_e is None:                      # first call: no derivative
            self.prev_e, self.prev_y = e, y
        raw_d = (-(y - self.prev_y) if self.d_on == "measurement"
                 else (e - self.prev_e)) / self.dt
        a = self.d_tau / (self.d_tau + self.dt)      # a = 0 -> no filtering
        self.d_filt = a * self.d_filt + (1 - a) * raw_d
        self.prev_e, self.prev_y = e, y

        p, d = self.kp * e, self.kd * self.d_filt
        u_unsat = p + self.integral + d
        u = float(np.clip(u_unsat, self.u_min, self.u_max))

        i_rate = self.ki * e
        if self.antiwindup == "backcalc":
            i_rate += self.kb * (u - u_unsat)        # bleeds integral while saturated
        self.integral += i_rate * self.dt
        self.last = (p, self.integral, d)
        return u


class SpringMassDamper:
    """m x'' + c x' + k x = u   (state: position, velocity). RK4 integration."""

    def __init__(self, m=1.0, c=10.0, k=20.0):
        self.m, self.c, self.k = m, c, k
        self.reset()

    def reset(self):
        self.s = np.zeros(2)

    def _f(self, s, u):
        return np.array([s[1], (u - self.c * s[1] - self.k * s[0]) / self.m])

    def step(self, u, dt):
        s, f = self.s, self._f
        k1 = f(s, u); k2 = f(s + dt / 2 * k1, u)
        k3 = f(s + dt / 2 * k2, u); k4 = f(s + dt * k3, u)
        self.s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def y(self):
        return self.s[0]


class ThirdOrderLag:
    """G(s) = 1/(s+1)^3 as three cascaded first-order lags (exact per-step update)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.s = np.zeros(3)

    def step(self, u, dt):
        a = np.exp(-dt)
        x = u
        for i in range(3):                           # input held constant over dt
            self.s[i] = a * self.s[i] + (1 - a) * x
            x = self.s[i]

    @property
    def y(self):
        return self.s[2]


def simulate(plant, controller, setpoint, T=3.0, dt=0.001, noise=0.0, seed=0,
             sensor_dt=None):
    """Closed-loop run. `setpoint` is a float or f(t). The plant is always
    integrated at the fine step `dt`; the controller runs every `ctrl_dt`
    (set controller.dt) with its output held in between (zero-order hold)."""
    rng = np.random.default_rng(seed)
    plant.reset(); controller.reset()
    sp_f = setpoint if callable(setpoint) else (lambda t: setpoint)
    n = int(round(T / dt))
    hold = max(1, int(round(controller.dt / dt)))
    t = np.arange(n) * dt
    y = np.zeros(n); u = np.zeros(n); r = np.zeros(n)
    P = np.zeros(n); I = np.zeros(n); D = np.zeros(n)
    u_now = 0.0
    for i in range(n):
        r[i] = sp_f(t[i])
        y[i] = plant.y
        if i % hold == 0:
            meas = y[i] + (rng.normal(0, noise) if noise else 0.0)
            u_now = controller.update(r[i], meas)
        u[i] = u_now
        P[i], I[i], D[i] = controller.last
        plant.step(u_now, dt)
    return dict(t=t, y=y, u=u, r=r, P=P, I=I, D=D)


def metrics(t, y, target=1.0, band=0.02):
    """Rise time (10-90 %), overshoot %, settling time (2 % band), steady-state error."""
    y = np.asarray(y)
    def first_cross(level):
        idx = np.where(y >= level)[0]
        return t[idx[0]] if len(idx) else np.nan
    t10, t90 = first_cross(0.1 * target), first_cross(0.9 * target)
    rise = t90 - t10 if not np.isnan(t90) else np.nan
    overshoot = max(0.0, (y.max() - target) / target * 100)
    outside = np.where(np.abs(y - target) > band * target)[0]
    if len(outside) == 0:
        settle = 0.0
    elif outside[-1] >= len(t) - 1:
        settle = np.nan                               # never settles in window
    else:
        settle = t[outside[-1] + 1]
    ss = target - y[-int(0.05 * len(y)):].mean()      # mean of last 5 % of the run
    return dict(rise=rise, overshoot=overshoot, settle=settle, sse=ss)
