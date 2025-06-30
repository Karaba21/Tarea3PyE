import pandas as pd
import numpy as np
from scipy.stats import chi2, t

# Cargar datos
muestra = pd.read_csv("muestra_ech.csv")
velocidad = pd.read_csv("velocidad_internet_ucu.csv")

# -------------------------------
# PROBLEMA 1: INGRESO POR DEPARTAMENTO
# -------------------------------

# 1. Ingreso per cápita
muestra["ingreso_pc"] = muestra["ingreso"] / muestra["personas_hogar"]

# 2. Clasificar en quintiles
quintiles = np.percentile(muestra["ingreso_pc"], [20, 40, 60, 80])

# 3. Filtrar top 20%
top_20 = muestra[muestra["ingreso_pc"] > quintiles[3]]

# 4. Frecuencia observada por departamento
obs = top_20["departamento"].value_counts().sort_index()

# 5. Frecuencia esperada (uniforme)
k = muestra["departamento"].nunique()
total_top = len(top_20)
exp = pd.Series([total_top / k] * k, index=range(1, k+1))

# 6. Estadístico chi-cuadrado
chi2_stat = (((obs - exp)**2) / exp).sum()

# 7. Valor crítico y decisión
chi2_crit = chi2.ppf(0.95, df=k-1)
p_val_chi2 = 1 - chi2.cdf(chi2_stat, df=k-1)

# -------------------------------
# PROBLEMA 2: VELOCIDAD INTERNET
# -------------------------------

# 1. Filtrar Central y Semprún
filtrada = velocidad[velocidad["Edificio"].isin(["Central", "Semprún"])]

# 2. Estadísticas descriptivas
media_c = filtrada[filtrada["Edificio"] == "Central"]["Velocidad Mb/s"].mean()
media_s = filtrada[filtrada["Edificio"] == "Semprún"]["Velocidad Mb/s"].mean()
sd_c = filtrada[filtrada["Edificio"] == "Central"]["Velocidad Mb/s"].std()
sd_s = filtrada[filtrada["Edificio"] == "Semprún"]["Velocidad Mb/s"].std()
n_c = filtrada[filtrada["Edificio"] == "Central"].shape[0]
n_s = filtrada[filtrada["Edificio"] == "Semprún"].shape[0]

# 3. Estadístico t
num = media_c - media_s
den = np.sqrt((sd_c**2)/n_c + (sd_s**2)/n_s)
t_stat = num / den

# 4. Grados de libertad (Welch)
df = ((sd_c**2/n_c + sd_s**2/n_s)**2) / (((sd_c**2/n_c)**2)/(n_c-1) + ((sd_s**2/n_s)**2)/(n_s-1))
p_val_t = t.cdf(t_stat, df=df)  # unilateral izquierda

# 5. Valor crítico t
crit_t = t.ppf(0.05, df=df)

# -------------------------------
# RESULTADOS
# -------------------------------

print("PROBLEMA 1: INGRESO POR DEPARTAMENTO")
print(f"Estadístico chi-cuadrado: {chi2_stat:.2f}")
print(f"Valor crítico (alfa=0.05): {chi2_crit:.2f}")
print(f"p-valor: {p_val_chi2:.4f}")
print("Se rechaza H0" if chi2_stat > chi2_crit else "No se rechaza H0")

print("\nPROBLEMA 2: VELOCIDAD DE INTERNET")
print(f"t estadístico: {t_stat:.2f}")
print(f"Valor crítico t (alfa=0.05): {crit_t:.2f}")
print(f"p-valor: {p_val_t:.4f}")
print("Se rechaza H0" if t_stat < crit_t else "No se rechaza H0")