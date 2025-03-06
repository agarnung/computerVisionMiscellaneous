import csv
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos de los archivos
with open('averages_results.txt', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    averages = list(reader)

avgOOPBadTime, avgOOPBadCycles = float(averages[0][0]), int(averages[0][1])
avgOOPDOPTime, avgOOPDOPCycles = float(averages[0][2]), int(averages[0][3])
avgOOPDOP_GoodWithFooPaddingTime, avgOOPDOP_GoodWithFooPaddingCycles = float(averages[0][4]), int(averages[0][5])

# Inicialización de las posiciones
position_1_oop_bad_ms = 0
position_2_oop_bad_ms = 0
position_3_oop_bad_ms = 0

position_1_oop_good_dop_ms = 0
position_2_oop_good_dop_ms = 0
position_3_oop_good_dop_ms = 0

position_1_oop_good_dop_foo_padding_ms = 0
position_2_oop_good_dop_foo_padding_ms = 0
position_3_oop_good_dop_foo_padding_ms = 0

position_1_oop_bad_ticks = 0
position_2_oop_bad_ticks = 0
position_3_oop_bad_ticks = 0

position_1_oop_good_dop_ticks = 0
position_2_oop_good_dop_ticks = 0
position_3_oop_good_dop_ticks = 0

position_1_oop_good_dop_foo_padding_ticks = 0
position_2_oop_good_dop_foo_padding_ticks = 0
position_3_oop_good_dop_foo_padding_ticks = 0

# Cargar los resultados de benchmark
with open('benchmark_results.txt', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)

for row in data:
    time_data = [float(row[1]), float(row[2]), float(row[3])]
    cycle_data = [int(row[4]), int(row[5]), int(row[6])]

    sorted_time_indices = np.argsort(time_data)
    if sorted_time_indices[0] == 0:
        position_1_oop_bad_ms += 1
    elif sorted_time_indices[0] == 1:
        position_1_oop_good_dop_ms += 1
    else:
        position_1_oop_good_dop_foo_padding_ms += 1
    
    if sorted_time_indices[1] == 0:
        position_2_oop_bad_ms += 1
    elif sorted_time_indices[1] == 1:
        position_2_oop_good_dop_ms += 1
    else:
        position_2_oop_good_dop_foo_padding_ms += 1
    
    if sorted_time_indices[2] == 0:
        position_3_oop_bad_ms += 1
    elif sorted_time_indices[2] == 1:
        position_3_oop_good_dop_ms += 1
    else:
        position_3_oop_good_dop_foo_padding_ms += 1

    sorted_cycle_indices = np.argsort(cycle_data)
    if sorted_cycle_indices[0] == 0:
        position_1_oop_bad_ticks += 1
    elif sorted_cycle_indices[0] == 1:
        position_1_oop_good_dop_ticks += 1
    else:
        position_1_oop_good_dop_foo_padding_ticks += 1
    
    if sorted_cycle_indices[1] == 0:
        position_2_oop_bad_ticks += 1
    elif sorted_cycle_indices[1] == 1:
        position_2_oop_good_dop_ticks += 1
    else:
        position_2_oop_good_dop_foo_padding_ticks += 1
    
    if sorted_cycle_indices[2] == 0:
        position_3_oop_bad_ticks += 1
    elif sorted_cycle_indices[2] == 1:
        position_3_oop_good_dop_ticks += 1
    else:
        position_3_oop_good_dop_foo_padding_ticks += 1

# Etiquetas de métodos
labels = ['OOP Bad Order', 'OOP Good DOP', 'OOP Good DOP with Foo Padding']

# Datos de las posiciones
position_1_ms = [position_1_oop_bad_ms, position_1_oop_good_dop_ms, position_1_oop_good_dop_foo_padding_ms]
position_2_ms = [position_2_oop_bad_ms, position_2_oop_good_dop_ms, position_2_oop_good_dop_foo_padding_ms]
position_3_ms = [position_3_oop_bad_ms, position_3_oop_good_dop_ms, position_3_oop_good_dop_foo_padding_ms]

position_1_ticks = [position_1_oop_bad_ticks, position_1_oop_good_dop_ticks, position_1_oop_good_dop_foo_padding_ticks]
position_2_ticks = [position_2_oop_bad_ticks, position_2_oop_good_dop_ticks, position_2_oop_good_dop_foo_padding_ticks]
position_3_ticks = [position_3_oop_bad_ticks, position_3_oop_good_dop_ticks, position_3_oop_good_dop_foo_padding_ticks]

# Ancho de las barras y posición de los elementos en el gráfico
bar_width = 0.2
index = np.arange(len(labels))

# Gráfico de milisegundos
fig, ax = plt.subplots(figsize=(12, 7))

bar1 = ax.bar(index - bar_width, position_1_ms, bar_width, color='gold', label='Gold', edgecolor='black', linewidth=1.2, zorder=5)
bar2 = ax.bar(index, position_2_ms, bar_width, color='silver', label='Silver', edgecolor='black', linewidth=1.2, zorder=4)
bar3 = ax.bar(index + bar_width, position_3_ms, bar_width, color='#cd7f32', label='Bronze', edgecolor='black', linewidth=1.2, zorder=3)

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency of medals', fontsize=14, fontweight='bold')
ax.set_title('Podium in seconds', fontsize=16, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Mostrar los promedios debajo de las barras (con más decimales)
ax.text(-0.5, -135, f'Average in seconds:', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
for i, avg in enumerate([avgOOPBadTime, avgOOPDOPTime, avgOOPDOP_GoodWithFooPaddingTime]):
    ax.text(i, -135, f'{avg:.6f}', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# Ajustar el gráfico
plt.subplots_adjust(top=0.9, bottom=0.15)
plt.savefig('podium_comparison_ms.png')

# Gráfico de ciclos de CPU
fig, ax = plt.subplots(figsize=(12, 7))

bar1 = ax.bar(index - bar_width, position_1_ticks, bar_width, color='gold', label='Gold', edgecolor='black', linewidth=1.2, zorder=5)
bar2 = ax.bar(index, position_2_ticks, bar_width, color='silver', label='Silver', edgecolor='black', linewidth=1.2, zorder=4)
bar3 = ax.bar(index + bar_width, position_3_ticks, bar_width, color='#cd7f32', label='Bronze', edgecolor='black', linewidth=1.2, zorder=3)

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency of medals', fontsize=14, fontweight='bold')
ax.set_title('Podium in CPU cycles', fontsize=16, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Mostrar los promedios de ciclos debajo de las barras
ax.text(-0.5, -135, f'Average in CPU cycles:', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
for i, avg in enumerate([avgOOPBadCycles, avgOOPDOPCycles, avgOOPDOP_GoodWithFooPaddingCycles]):
    ax.text(i, -135, f'{avg}', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# Ajustar el gráfico
plt.subplots_adjust(top=0.9, bottom=0.15)
plt.savefig('podium_comparison_ticks.png')

