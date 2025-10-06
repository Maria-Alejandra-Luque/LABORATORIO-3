# LABORATORIO-3
## DESCRIPCIÓN 
En este repositorio analizaremos la practica desarollada que se titula **"Análisis espectral de la voz"**, donde buscamos entender las principales características de la voz humana desde el punto de vista del procesamiento digital de señales. Haciendo uso de varias grabaciones de voces masculinas y femeninas y se trabajó con ellas en Python, aplicando herramientas como la Transformada de Fourier para observar su comportamiento en el dominio de la frecuencia.

El propósito general fue comparar las diferencias entre voces de hombres y mujeres, tanto a nivel espectral como en sus características vocales, y relacionar estos resultados con aspectos biomédicos y de Ingeniería comprendiendo mejor el comportamiento de la voz.

## OBJETIVOS
- Capturar y procesar señales de voz masculinas y femeninas.<br>
- Aplicar la Transformada de Fourier como herramienta de análisis espectral de la voz.<br>
- Extraer parámetros característicos de la señal de voz: frecuencia fundamental,frecuencia media, brillo, intensidad, jitter y shimmer.<br>
- Comparar las diferencias principales entre señales de voz de hombres y mujeres a partir de su análisis en frecuencia.<br>
- Desarrollar conclusiones sobre el comportamiento espectral de la voz humana
en función del género. <br>

# PROCEDIMIENTO
## PARTE A- Adquisición de las señales de voz
En esta primera parte del laboratorio se realizó la grabación de las señales de voz. Obtuvimos las muestras de audio de diferentes personas (hombres y mujeres) pronunciando la misma frase corta ("Un simple eco es la prueba de que el aire guarda memorias, y que la música de la ciencia también puede acariciar el alma.”), garantizando que todas las grabaciones tuvieran las mismas condiciones de muestreo para permitir una comparación mas precisa.<br>

Las grabaciones se realizarón con micrófonos de teléfonos, cuidando que no hubiera ruido externo ni saturación durante la captura. Cada archivo se guardó en formato .wav. Posteriormente, las señales fueron importadas a Python, donde se graficaron en el dominio del tiempo y se prepararon para el análisis espectral.<br>
## Diagrama 
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/e3bf8b67-7b65-41e7-adb7-0d5adea2ad00" /> <br>

### Archivos de Audio

**Hombre 1:**

https://github.com/user-attachments/assets/a866c77e-11af-423b-9f67-2854d0a0ca18

 <br>
 
**Hombre 2:**

https://github.com/user-attachments/assets/fdc8c338-f863-41ed-a14e-bc15ec328939

 <br>
 
**Hombre 3:**

https://github.com/user-attachments/assets/34fd4192-5114-49ed-964c-2844d8378798

<br>

**MUJER 1:**

https://github.com/user-attachments/assets/83107f68-9953-410b-9e63-d19764eaf8b2

 <br>
 
**MUJER 2:** 

https://github.com/user-attachments/assets/5339be7d-be98-4224-8050-b320b0845c84

 <br>
 
**MUJER 3:** 

https://github.com/user-attachments/assets/fb98c625-09d1-40ed-a1fe-1a8601a72c8f

<br>

### CODIGO

```
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
carpeta = "/content/drive/MyDrive/Colab Notebooks"
rutas = glob.glob(os.path.join(carpeta, "*.wav"))
print("Archivos encontrados:")
for r in rutas:
    print(r)

def calcular_caracteristicas(data, samplerate):
    """ Calcula las características pedidas: F0, Fmedia, Brillo, Energía """

    # FFT
    N = len(data)
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, 1/samplerate)

    # Solo frecuencias positivas
    idx = np.where(freqs > 0)
    freqs = freqs[idx]
    magnitudes = np.abs(fft_data[idx])

    # Frecuencia fundamental = pico máximo (excluyendo DC)
    f0 = freqs[np.argmax(magnitudes[1:]) + 1]

    # Frecuencia media (ponderada por magnitud)
    f_media = np.sum(freqs * magnitudes) / np.sum(magnitudes)

    # Brillo: proporción de energía > 1500 Hz
    energia_total = np.sum(magnitudes**2)
    energia_alta = np.sum(magnitudes[freqs > 1500]**2)
    brillo = energia_alta / energia_total if energia_total > 0 else 0

    # Intensidad (energía total en el tiempo)
    intensidad = np.sum(data**2)
    return f0, f_media, brillo, intensidad, freqs, magnitudes
```

En este fragmento de código se leen los archivos de voz en formato .wav y se calcula su contenido en frecuencia usando la Transformada Rápida de Fourier (FFT). La función calcular_caracteristicas() toma como entrada los datos del audio y su frecuencia de muestreo, y obtiene diferentes parámetros de la señal. <br>
Primero, genera el espectro de frecuencias y selecciona solo las componentes positivas. Luego, identifica el pico de mayor magnitud para determinar la frecuencia fundamental (F0). Posteriormente, calcula la frecuencia media o centroide espectral ponderando las magnitudes, y evalúa el brillo, que corresponde a la proporción de energía ubicada por encima de 1500 Hz. Finalmente, obtiene la intensidad como la energía total de la señal en el tiempo. Estos resultados permiten caracterizar cada voz y compararlas según sus propiedades acústicas.<br>


```
resultados = []
for ruta in rutas:
    # Cargar audio
    data, samplerate = sf.read(ruta)

    # Si es estéreo -> a mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Tiempo
    tiempo = np.linspace(0, len(data)/samplerate, num=len(data))
 # Calcular características
    f0, f_media, brillo, intensidad, freqs, magnitudes = calcular_caracteristicas(data, samplerate)

    # Guardar resultados en tabla
    resultados.append({
        "Archivo": os.path.basename(ruta),
        "Frecuencia Fundamental (Hz)": f0,
        "Frecuencia Media (Hz)": f_media,
        "Brillo": brillo,
        "Intensidad (Energía)": intensidad
    })

    # Graficar señal + espectro
    plt.figure(figsize=(14,5))

    # Señal en el tiempo
    plt.subplot(1,2,1)
    plt.plot(tiempo, data, color="fuchsia")
    plt.title(f"Señal en el tiempo - {os.path.basename(ruta)}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    # Espectro
    plt.subplot(1,2,2)
    plt.plot(freqs, magnitudes, color="fuchsia")
    plt.title(f"Espectro de frecuencia - {os.path.basename(ruta)}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.xlim(0, 2000)  # límite a 2 kHz

    plt.tight_layout()
    plt.show()

df = pd.DataFrame(resultados)
print("\nTABLA DE RESULTADOS (Parte A):")
print(df)
```
## PARTE B


## DESCRIPCIÓN 
En esta parte del trabajo se realizó la medición del Jitter y el Shimmer a partir de grabaciones de voz masculina y femenina. Se aplicó un filtro pasa-banda butterworth para eliminar el ruido no deseado y luego se analizaron las señales para identificar los periodos de vibración y los picos de amplitud. Con estos datos se calcularon los valores absolutos y relativos de Jitter (variación en la frecuencia fundamental) y Shimmer (variación en la amplitud), utilizando las fórmulas correspondientes. Finalmente, se presentaron los resultados obtenidos para tres voces masculinas y tres femeninas.
# Diagrama 
<img width="1024" height="768" alt="(PARTE B)" src="https://github.com/user-attachments/assets/b2085e87-d0bc-498a-9197-98db7b7caeb1" />

# PROCEDIMIENTO
En la primera parte del código se realiza el montaje del entorno de trabajo en Google Colab y la definición de los archivos de voz que serán analizados. Mediante el comando drive.mount() se conecta Google Drive para acceder a las grabaciones almacenadas, estableciendo una ruta base donde se ubican los archivos .wav. A continuación, se crea un diccionario llamado archivos que contiene la información de cada muestra de voz —nombre del archivo, rango de frecuencias de paso y orden del filtro Butterworth— diferenciando entre voces masculinas y femeninas. Esta etapa es fundamental porque organiza los datos de entrada y define los parámetros iniciales con los que posteriormente se aplicará el filtrado y el análisis de Jitter y Shimmer.



## PARTE C  Analisis Comparativo
En la ultima parte se compararon los parámetros acústicos obtenidos en las partes A y B para identificar las diferencias más relevantes entre voces masculinas y femeninas.
Los parámetros analizados fueron:

Frecuencia fundamental (Hz)

Frecuencia media (Hz)

Brillo (proporción de energía en frecuencias altas >1500 Hz)

Intensidad (energía total)

Jitter (variación de frecuencia)

Shimmer (variación de amplitud)

<img width="1024" height="768" alt="Diseño sin título" src="https://github.com/user-attachments/assets/5686443e-b937-475a-8d7f-f7a0bed3a210" />

# CODIGO
```

PARTE C: ANÁLISIS COMPARATIVO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from difflib import get_close_matches


 #1️ CARGAR RESULTADOS DE A Y B


rutaA = "/content/drive/MyDrive/Colab Notebooks/resultados_parteA.csv"
rutaB = "/content/drive/MyDrive/LaboratorioVoz/resultados_parteB/resultados_parteB.csv"

parteA = pd.read_csv(rutaA)
parteB = pd.read_csv(rutaB)


# 2️ NORMALIZAR Y EMPAREJAR NOMBRES

parteA["Archivo_limpio"] = (
    parteA["Archivo"].str.lower().str.replace(".wav", "").str.replace(" ", "").str.strip()
)
parteB["Voz_limpio"] = (
    parteB["Voz"].str.lower().str.replace(".wav", "").str.replace(" ", "").str.strip()
)

matches = {}
for a in parteA["Archivo_limpio"]:
    m = get_close_matches(a, parteB["Voz_limpio"], n=1, cutoff=0.5)
    if m:
        matches[a] = m[0]


# 3️ UNIR LOS RESULTADOS DE A Y B


merged_rows = []
for i, row in parteA.iterrows():
    a_clean = row["Archivo_limpio"]
    if a_clean in matches:
        b_match = parteB[parteB["Voz_limpio"] == matches[a_clean]].iloc[0]
        merged_rows.append({
            "Archivo": row["Archivo"],
            "Frecuencia Fundamental (Hz)": row["Frecuencia Fundamental (Hz)"],
            "Frecuencia Media (Hz)": row["Frecuencia Media (Hz)"],
            "Brillo": row["Brillo"],
            "Intensidad (Energía)": row["Intensidad (Energía)"],
            "Jitter_abs (s)": b_match["Jitter_abs (s)"],
            "Jitter_rel (%)": b_match["Jitter_rel (%)"],
            "Shimmer_abs": b_match["Shimmer_abs"],
            "Shimmer_rel (%)": b_match["Shimmer_rel (%)"]
        })

df = pd.DataFrame(merged_rows)


# 4 CLASIFICAR POR GÉNERO


df["Género"] = np.where(df["Archivo"].str.contains("mujer", case=False), "Mujer", "Hombre")


# 5️ CALCULAR PROMEDIOS POR GÉNERO


promedios = df.groupby("Género")[[
    "Frecuencia Fundamental (Hz)",
    "Frecuencia Media (Hz)",
    "Brillo",
    "Intensidad (Energía)",
    "Jitter_rel (%)",
    "Shimmer_rel (%)"
]].mean().reset_index()

print("\n===== TABLA COMBINADA =====")
print(df)
print("\n===== PROMEDIOS POR GÉNERO =====")
print(promedios)


# GRAFICAR COMPARACIONES


fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Comparación de Parámetros de Voz entre Hombres y Mujeres", fontsize=14)

# Títulos más claros y etiquetas mejoradas
parametros = [
    ("Frecuencia Fundamental (Hz)", "Frecuencia base (Hz)"),
    ("Frecuencia Media (Hz)", "Frecuencia promedio (Hz)"),
    ("Brillo", "Proporción de energía (>1500 Hz)"),
    ("Intensidad (Energía)", "Nivel de energía total"),
    ("Jitter_rel (%)", "Variación frecuencia (%)"),
    ("Shimmer_rel (%)", "Variación amplitud (%)")
]

for ax, (param, ylabel) in zip(axes.flatten(), parametros):
    ax.bar(promedios["Género"], promedios[param],
           color=["blue", "fuchsia"], alpha=0.8)
    ax.set_title(param)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Género")
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# 7️ GUARDAR RESULTADOS FINALES


out_path = "/content/drive/MyDrive/LaboratorioVoz/resultados_parteC.csv"
promedios.to_csv(out_path, index=False)
print(f"\n Parte C completada. Resultados guardados en: {out_path}")
```

### DESCRIPCIÓN CODIGO

Carga de resultados
Se importaron los archivos CSV generados en las partes A y B con los parámetros espectrales e inestabilidades vocales.

Normalización de nombres
Se limpiaron los nombres de los archivos (.wav) para emparejar correctamente los registros de ambas partes.

Unión de datos
Se fusionaron ambos conjuntos (Parte A + Parte B) en un solo DataFrame con todos los parámetros de cada archivo de voz.

Clasificación por género
El código detecta si el nombre del archivo contiene “mujer” o “hombre” y asigna la categoría correspondiente.

Cálculo de promedios
Se agrupan los datos por género y se calculan los valores promedio para cada parámetro.

Visualización
Se generan gráficos comparativos (barras) que muestran de manera visual las diferencias entre hombres y mujeres.

Exportación
Los resultados finales se guardan en resultados_parteC.csv.


### PREGUNTAS LABORATORIO
Gracias a los resultados obtenidos , se pudo dar respuesta a las siguientes preguntas:
1. ¿Qué diferencias se observan en la frecuencia fundamental?

La frecuencia fundamental (F₀) fue mayor en las voces masculinas (≈454 Hz) que en las femeninas (≈314 Hz).
Esto muestra que las voces de los hombres tienen una frecuencia base más alta , aunque normalmente se espera lo contrario (voces masculinas con menor F₀).
La diferencia puede deberse a la frase usada, tono de emisión o variaciones en la grabación.

2. ¿Qué otras diferencias se notan en términos de brillo, media o intensidad?

Frecuencia media: Las voces femeninas presentan un valor mayor (~4526 Hz) frente a los hombres (~4017 Hz). Esto indica una mayor concentración de energía en frecuencias altas, lo que se percibe como una voz más aguda y clara.

Brillo: También es ligeramente superior en las mujeres (0.1099 frente a 0.1060), lo cual refuerza la presencia de armónicos altos en su espectro.

Intensidad (Energía): Las voces masculinas tienen una energía promedio mayor (~4890 vs 4153), lo que puede relacionarse con una mayor proyección vocal.

En resumen, las mujeres muestran voces más agudas y brillantes, mientras que los hombres presentan voces más potentes o intensas.

3. Conclusiones sobre el comportamiento de la voz en hombres y mujeres

Las voces femeninas tienden a presentar una frecuencia media y brillo mayores, generando un timbre más agudo y con más contenido en altas frecuencias.

Las voces masculinas presentan mayor energía e intensidad, lo que produce un sonido más fuerte y grave.

Los parámetros de jitter y shimmer fueron ligeramente más altos en hombres, lo que sugiere una mayor variabilidad tanto en frecuencia como en amplitud.

En general, las diferencias espectrales observadas son coherentes con las características anatómicas y fisiológicas de cada género: las cuerdas vocales masculinas son más largas y gruesas, lo que normalmente genera frecuencias fundamentales más bajas.

4. Importancia clínica del jitter y shimmer en el análisis de la voz

El jitter y el shimmer son indicadores de la estabilidad vocal:

El jitter mide la variación ciclo a ciclo de la frecuencia (irregularidad temporal).

El shimmer mide la variación en la amplitud de los ciclos (irregularidad de intensidad).

En el ámbito clínico, valores elevados pueden indicar alteraciones en el control de la fonación, como disfonías, nódulos vocales, parálisis o fatiga de las cuerdas vocales.
Por tanto, estos parámetros son fundamentales para el diagnóstico de patologías vocales, seguimiento de terapias y evaluación del rendimiento vocal en profesionales de la voz.
