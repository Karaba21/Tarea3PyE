import pandas as pd
import numpy as np
from scipy.stats import chi2, t

# Cargar datos
muestra = pd.read_csv("muestra_ech.csv")
velocidad = pd.read_csv("velocidad_internet_ucu.csv")

# -------------------------------
# PROBLEMA 1: INGRESO POR DEPARTAMENTO
# -------------------------------

# 1. Calcular el ingreso per capita de cada hogar.
muestra["ingreso_pc"] = muestra["ingreso"] / muestra["personas_hogar"]

# 2. Clasificar todos los hogares en quintiles segun el ingreso per capita usando percentiles
quintiles = np.percentile(muestra["ingreso_pc"], [20, 40, 60, 80])

# 3. Filtrar los hogares que pertenecen al quintil superior (20 % con mayor ingreso).
top_20 = muestra[muestra["ingreso_pc"] > quintiles[3]]

# 4. Construir una tabla de frecuencias observadas de hogares de ingreso alto por departamento.
obs = top_20["departamento"].value_counts().sort_index()

# 5. Calcular las frecuencias esperadas bajo la hipotesis de distribucion uniforme.
k = muestra["departamento"].nunique()
total_top = len(top_20)
exp = pd.Series([total_top / k] * k, index=range(1, k+1))

# 6. Calcular el estadístico chi-cuadrado.
chi2_stat = (((obs - exp)**2) / exp).sum()

# 7. Determinar si se rechaza o no la hipotesis nula con α = 0,05 y k = 19.
chi2_crit = chi2.ppf(0.95, df=k-1)
p_val_chi2 = 1 - chi2.cdf(chi2_stat, df=k-1)

# -------------------------------
# PROBLEMA 2: VELOCIDAD INTERNET
# -------------------------------

# 1. Filtrar del conjunto de datos las observaciones correspondientes a UCU (Central) y UCU (Semprun).
filtrada = velocidad[velocidad["Edificio"].isin(["Central", "Semprún"])]

# 2. Calcular la media, desviacion estandar y tamano muestral de la velocidad para cada edificio
media_c = filtrada[filtrada["Edificio"] == "Central"]["Velocidad Mb/s"].mean()
media_s = filtrada[filtrada["Edificio"] == "Semprún"]["Velocidad Mb/s"].mean()
sd_c = filtrada[filtrada["Edificio"] == "Central"]["Velocidad Mb/s"].std()
sd_s = filtrada[filtrada["Edificio"] == "Semprún"]["Velocidad Mb/s"].std()
n_c = filtrada[filtrada["Edificio"] == "Central"].shape[0]
n_s = filtrada[filtrada["Edificio"] == "Semprún"].shape[0]

# 3. Aplicar la formula del estadistico t para comparar ambas medias.
num = media_c - media_s
den = np.sqrt((sd_c**2)/n_c + (sd_s**2)/n_s)
t_stat = num / den

# 4. Calcular o reportar el p-valor asociado al estadistico t.
df = ((sd_c**2/n_c + sd_s**2/n_s)**2) / (((sd_c**2/n_c)**2)/(n_c-1) + ((sd_s**2/n_s)**2)/(n_s-1))
p_val_t = t.cdf(t_stat, df=df)  # unilateral izquierda

# 5. Determinar si se rechaza o no la hipotesis nula con α = 0,05.
crit_t = t.ppf(0.05, df=df)

# -------------------------------
# RESULTADOS
# -------------------------------

print("####################################")
print("PROBLEMA 1: INGRESO POR DEPARTAMENTO")
print("####################################")
# Punto 1
print("\n1. Se calculó el ingreso per cápita de cada hogar como ingreso / personas en el hogar.")
# Punto 2
print("2. Se clasificaron todos los hogares en quintiles usando percentiles del ingreso per cápita.")
print(f"   Quintiles calculados (percentiles 20, 40, 60, 80): {quintiles}")
# Punto 3
print(f"3. Se filtraron los hogares que pertenecen al quintil superior (20% más ricos).")
print(f"   Total hogares en el quintil superior: {len(top_20)}")
# Punto 4
print("4. Frecuencia observada de hogares ricos por departamento:")
print("\nDepartamento | Observados")
for depto, freq in obs.items():
    print(f"{depto:>11} | {freq}")
# Punto 5
print("5. Frecuencia esperada bajo hipótesis de distribución uniforme:")
print(exp)
# Punto 6
print("6. Estadístico chi-cuadrado calculado:")
print(f"   chi-cuadrado = {chi2_stat:.2f}")
# Punto 7
print("7. Comparación con valor crítico para alfa = 0.05:")
print(f"   Valor crítico (gl = {k-1}): {chi2_crit:.2f}")
print(f"   p-valor: {p_val_chi2:.4f}")

print(f"Estadístico chi-cuadrado: {chi2_stat:.2f}")
print(f"Valor crítico (alfa=0.05): {chi2_crit:.2f}")
print(f"p-valor: {p_val_chi2:.4f}")
print("Se rechaza H0" if chi2_stat > chi2_crit else "No se rechaza H0")
print("\n")
print("####################################")
print("PROBLEMA 2: VELOCIDAD DE INTERNET")
print("####################################")

# Punto 1
print("\n1. Se filtraron las observaciones correspondientes a los edificios UCU (Central) y UCU (Semprún).")
print(f"   Total observaciones en Central: {n_c}")
print(f"   Total observaciones en Semprún: {n_s}")
# Punto 2
print("\n2. Estadísticas descriptivas:")
print(f"   Media velocidad Central: {media_c:.2f} Mb/s")
print(f"   Desviación estándar Central: {sd_c:.2f}")
print(f"   Media velocidad Semprún: {media_s:.2f} Mb/s")
print(f"   Desviación estándar Semprún: {sd_s:.2f}")
# Punto 3
print("\n3. Estadístico t calculado para la comparación de medias:")
print(f"   t = {t_stat:.4f}")
# Punto 4
print("\n4. p-valor asociado al estadístico t (prueba unilateral izquierda):")
print(f"   p-valor = {p_val_t:.4f}")
# Punto 5
print("\n5. Comparación con el valor crítico t para alfa = 0.05:")
print(f"   Grados de libertad aproximados (Welch): {df:.2f}")
print(f"   Valor crítico t (alfa = 0.05): {crit_t:.4f}")
if t_stat < crit_t:
    print("   🔴 Se rechaza la hipótesis nula: la velocidad promedio en Central es significativamente menor que en Semprún.")
else:
    print("   🟢 No se rechaza la hipótesis nula: no hay evidencia suficiente para afirmar que la velocidad en Central sea menor.")

print(f"t estadístico: {t_stat:.2f}")
print(f"Valor crítico t (alfa=0.05): {crit_t:.2f}")
print(f"p-valor: {p_val_t:.4f}")
print("Se rechaza H0" if t_stat < crit_t else "No se rechaza H0")