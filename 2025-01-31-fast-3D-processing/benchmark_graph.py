import csv
import matplotlib.pyplot as plt
import numpy as np

color_1st_place = 'gold'
color_2nd_place = 'silver'
color_3rd_place = '#cd7f32' # bronce
color_4th_place = '#B27333' # cobre
color_5th_place = '#6F7474' # nickel
color_6th_place = '#563622' # barro

# Cargar datos de los archivos
with open('averages_results.txt', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    averages = list(reader)

# Promedios de tiempo
avgOpenCVDirectTime = float(averages[0][0])
avgOpenCVPtrTime = float(averages[0][1])
avgOpenCVPtrVecTime = float(averages[0][2])
avgEigenTime = float(averages[0][3])
avgParallelTime = float(averages[0][4])
avgSIMDTime = float(averages[0][5])

# Inicialización de las posiciones para 6 medallas
positions = {
    "OpenCV\ndirect": [0, 0, 0, 0, 0, 0],
    "OpenCV\ndouble ptr": [0, 0, 0, 0, 0, 0],
    "OpenCV\nVec3d ptr": [0, 0, 0, 0, 0, 0],
    "Eigen": [0, 0, 0, 0, 0, 0],
    "Parallel": [0, 0, 0, 0, 0, 0],
    "SIMD": [0, 0, 0, 0, 0, 0]
}

# Cargar los resultados de benchmark (ignorando la primera columna)
with open('benchmark_results.txt', 'r') as file:
    reader = csv.reader(file)
    header = next(reader) # Ignorar la primera fila (encabezados)
    data = list(reader)

# Asignar posiciones basado en los tiempos
for row in data:
    time_data = [
        float(row[1]), # OpenCV directo
        float(row[2]), # OpenCV puntero double
        float(row[3]), # OpenCV puntero Vec3d
        float(row[4]), # Eigen
        float(row[5]), # Paralelo
        float(row[6])  # SIMD
    ]
    
    # Ordenar los tiempos y obtener los índices
    sorted_time_indices = np.argsort(time_data)  # Obtener los índices de los tiempos ordenados
    
    # Asignar posiciones según los tiempos ordenados
    positions[list(positions.keys())[sorted_time_indices[0]]][0] += 1 # 1er puesto
    positions[list(positions.keys())[sorted_time_indices[1]]][1] += 1 # 2do puesto
    positions[list(positions.keys())[sorted_time_indices[2]]][2] += 1 # 3er puesto
    positions[list(positions.keys())[sorted_time_indices[3]]][3] += 1 # 4to puesto
    positions[list(positions.keys())[sorted_time_indices[4]]][4] += 1 # 5to puesto
    positions[list(positions.keys())[sorted_time_indices[5]]][5] += 1 # 6to puesto

# Etiquetas de métodos
labels = ['OpenCV\ndirect', 'OpenCV\ndouble ptr', 'OpenCV\nVec3d ptr', 'Eigen', 'Parallel', 'SIMD']

# Datos de las posiciones
position_1 = [positions[label][0] for label in labels]
position_2 = [positions[label][1] for label in labels]
position_3 = [positions[label][2] for label in labels]
position_4 = [positions[label][3] for label in labels]
position_5 = [positions[label][4] for label in labels]
position_6 = [positions[label][5] for label in labels]

# Ancho de las barras y posición de los elementos en el gráfico
bar_width = 0.1
index = np.arange(len(labels))

# Gráfico de tiempos
fig, ax = plt.subplots(figsize=(15, 7))

bar1 = ax.bar(index - 2.5 * bar_width, position_1, bar_width, color=color_1st_place, label='1st place', edgecolor='black', linewidth=1.2, zorder=5)
bar2 = ax.bar(index - 1.5 * bar_width, position_2, bar_width, color=color_2nd_place, label='2nd place', edgecolor='black', linewidth=1.2, zorder=4)
bar3 = ax.bar(index - 0.5 * bar_width, position_3, bar_width, color=color_3rd_place, label='3rd place', edgecolor='black', linewidth=1.2, zorder=3)
bar4 = ax.bar(index + 0.5 * bar_width, position_4, bar_width, color=color_4th_place, label='4th place', edgecolor='black', linewidth=1.2, zorder=2)
bar5 = ax.bar(index + 1.5 * bar_width, position_5, bar_width, color=color_5th_place, label='5th place', edgecolor='black', linewidth=1.2, zorder=1)
bar6 = ax.bar(index + 2.5 * bar_width, position_6, bar_width, color=color_6th_place, label='6th place', edgecolor='black', linewidth=1.2, zorder=0)

ax.set_xlabel('Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency of medals', fontsize=14, fontweight='bold')
ax.set_title('Podium in seconds', fontsize=16, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Mostrar los promedios debajo de las barras (con más decimales)
ax.text(-0.8, -1845, f'Average in seconds:', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
for i, avg in enumerate([avgOpenCVDirectTime, avgOpenCVPtrTime, avgOpenCVPtrVecTime, avgEigenTime, avgParallelTime, avgSIMDTime]):
    ax.text(i, -1845, f'{avg:.6f}', ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

# Ajustar el gráfico
plt.subplots_adjust(top=0.9, bottom=0.15)
plt.savefig('podium_comparison_ms.png', dpi=700)

