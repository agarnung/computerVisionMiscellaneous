from pyomo.environ import *

# Crear modelo
model = ConcreteModel()

# Variables de decisión: número de platos (reales, >= 0)
platos = ['P1', 'P2', 'P3', 'P4']

# Parámetros: ganancias por plato
ganancia = {'P1': 6, 'P2': 4, 'P3': 7, 'P4': 5}

# Parámetros: uso de recursos por plato
ingredientes = {'P1': 0.3, 'P2': 0.6, 'P3': 0.75, 'P4': 0.5}
tiempo       = {'P1': 0.75, 'P2': 0.5, 'P3': 0.25, 'P4': 1}
pan          = {'P1': 0, 'P2': 2, 'P3': 1, 'P4': 1}

# Restricciones disponibles
max_ingredientes = 40
max_salsa = 30
max_tiempo = 10

# Mínimos requeridos por plato
minimos = {'P1': 2, 'P2': 4, 'P3': 6, 'P4': 3}

# Variables de decisión (con mínimos personalizados)
model.x = Var(platos, bounds=lambda model, p: (minimos[p], None), domain=Integers)

# Función objetivo: maximizar ganancias
model.obj = Objective(expr=sum(ganancia[p] * model.x[p] for p in platos), sense=maximize)

# Restricción de ingredientes
model.r_ingredientes = Constraint(expr=sum(ingredientes[p] * model.x[p] for p in platos) <= max_ingredientes)

# Restricción de tiempo
model.r_tiempo = Constraint(expr=sum(tiempo[p] * model.x[p] for p in platos) <= max_tiempo)

# Restricción de salsa especial
model.r_salsa = Constraint(expr=sum(salsa[p] * model.x[p] for p in platos) <= max_salsa)

# Resolver con solver
solver = SolverFactory('glpk') # O 'cbc' o 'gurobi' si están instalados
results = solver.solve(model)

print("Resultado de optimización:")
for p in platos:
    print(f"{p}: {model.x[p]():.0f} platos")

print(f"Ganancia total: {model.obj():.2f} €")

# Verificar que el solver reporta solución óptima
print(results.solver.status) # Debe mostrar 'ok'
print(results.solver.termination_condition)  # Debe mostrar 'optimal'

###########################
# Comprobar que es óptima #
###########################
import itertools

# Solución óptima obtenida con solver
opt_solution = {p: model.x[p]() for p in platos}
opt_value = model.obj()

print("Solución óptima del solver:", opt_solution)
print(f"Ganancia óptima del solver: {opt_value:.2f} €")

# Definir rango para cada plato: desde mínimo hasta mínimo + 10 (o un rango razonable)
rango_posibles = {p: range(minimos[p], minimos[p] + 11) for p in platos}

max_ganancia = -1
mejor_comb = None

for combo in itertools.product(rango_posibles['P1'], rango_posibles['P2'], rango_posibles['P3'], rango_posibles['P4']):
    # Asignar valores
    x_vals = dict(zip(platos, combo))

    # Comprobar restricciones
    ing_total = sum(ingredientes[p] * x_vals[p] for p in platos)
    tiempo_total = sum(tiempo[p] * x_vals[p] for p in platos)
    salsa_total = sum(salsa[p] * x_vals[p] for p in platos)

    if ing_total <= max_ingredientes and tiempo_total <= max_tiempo and salsa_total <= max_salsa:
        gan_total = sum(ganancia[p] * x_vals[p] for p in platos)
        if gan_total > max_ganancia:
            max_ganancia = gan_total
            mejor_comb = x_vals

print("\nMejor combinación encontrada por búsqueda exhaustiva:", mejor_comb)
print(f"Ganancia máxima encontrada: {max_ganancia:.2f} €")

# Comparar con la solución del solver
if abs(max_ganancia - opt_value) < 1e-5:
    print("La solución del solver es óptima.")
else:
    print("¡Encontramos una solución mejor! La solución del solver NO es óptima.")
