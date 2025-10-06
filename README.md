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
## PARTE C
