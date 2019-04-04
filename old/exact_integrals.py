
import sympy as sp


t, x, tp, e1, e2 = sp.symbols('t x tp e1 e2', real=True)

sol = sp.integrate(sp.I * sp.exp(-sp.I * e1 * (t-x) - sp.I * e2 * (x-tp)) * sp.I , (x, t, tp))

print(sol)
