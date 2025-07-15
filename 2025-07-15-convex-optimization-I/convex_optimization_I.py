from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

# Datos
A = [
    [0.80, 0.25],  # coeficientes: solar, eólica
    [0.85, 0.30],
    [0.90, 0.35],
]
y = [95, 110, 120]  # demanda observada (kWh)

n_periods = len(A)
n_sources = len(A[0])

# Modelo
model = ConcreteModel()

# Variables
model.x = Var(range(n_sources), bounds=[(0, 150), (0, 100)])  # solar hasta 150, eólica hasta 100
model.e = Var(range(n_periods), domain=NonNegativeReals)  # errores absolutos

# Objetivo: minimizar error total
model.obj = Objective(expr=sum(model.e[i] for i in range(n_periods)), sense=minimize)

# Restricciones
model.constraints = ConstraintList()
for i in range(n_periods):
    prod = sum(A[i][j] * model.x[j] for j in range(n_sources))
    model.constraints.add(model.e[i] >= prod - y[i])
    model.constraints.add(model.e[i] >= -(prod - y[i]))

# Resolver
solver = SolverFactory('glpk')
solver.solve(model)

# Resultados
x_opt = [value(model.x[j]) for j in range(n_sources)]
e_vals = [value(model.e[i]) for i in range(n_periods)]
total_error = sum(e_vals)

print("Límites de generación:")
print(f"  Solar (x[0]): 0 ≤ x ≤ 150 kWh")
print(f"  Eólica (x[1]): 0 ≤ x ≤ 100 kWh\n")

print(f"Óptimo encontrado (kWh): Solar = {x_opt[0]:.2f}, Eólica = {x_opt[1]:.2f}")
print(f"Error total: {total_error:.2f}")

# Visualización
lim_solar, lim_eolica = 150, 100
solar_vals = np.linspace(0, lim_solar, 100)
eolic_vals = np.linspace(0, lim_eolica, 100)
solar_grid, eolic_grid = np.meshgrid(solar_vals, eolic_vals)

def total_error(x_solar, x_eolic):
    x = np.array([x_solar, x_eolic])
    return sum(abs(A[i] @ x - y[i]) for i in range(n_periods))

Z = np.vectorize(total_error)(solar_grid, eolic_grid)

plt.figure(figsize=(10, 6))
contour = plt.contourf(solar_grid, eolic_grid, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Error absoluto total')
plt.plot(x_opt[0], x_opt[1], 'ro', label='Óptimo')
plt.xlabel('Energía solar (kWh)')
plt.ylabel('Energía eólica (kWh)')
plt.title('Optimización de generación energética')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./error_total_generacion.png")
print("Gráfico guardado como error_total_generacion.png")