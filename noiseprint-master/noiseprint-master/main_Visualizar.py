import numpy as np
import matplotlib.pyplot as plt
import os

def visualizar_prnu(ruta_npy, ruta_salida):
    print(f"Cargando PRNU desde: {ruta_npy}")
    matriz = np.load(ruta_npy)
    
    media = np.mean(matriz)
    desviacion = np.std(matriz)
    
    plt.figure(figsize=(6, 6))
    # Usamos escala de grises y limitamos el rango a +/- 3 desviaciones estándar
    plt.imshow(matriz, cmap='gray', vmin=media - 3*desviacion, vmax=media + 3*desviacion)
    plt.axis('off') 
    plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"✅ Imagen PRNU guardada en: {ruta_salida}")

def visualizar_noiseprint(ruta_npz, ruta_salida):
    print(f"Cargando Noiseprint desde: {ruta_npz}")
    datos = np.load(ruta_npz)
    matriz = datos['noiseprint'] # Sacamos la matriz del diccionario
    
    plt.figure(figsize=(6, 6))
    # Usamos un mapa de calor
    plt.imshow(matriz, cmap='viridis') 
    plt.axis('off')
    plt.savefig(ruta_salida, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"✅ Imagen Noiseprint guardada en: {ruta_salida}")

# ==================================================
# MAIN para visualizar las huellas PRNU y Noiseprint
# ==================================================
if __name__ == '__main__':
    # 1. Definimos y aseguramos la existencia de la carpeta de destino
    carpeta_destino = os.path.join("TFG", "visualizacion")
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # 2. Rutas de los archivos de entrada reales
    archivo_prnu = "TFG/huellas/huellasPRNU/iphone15/IMG_7887.npy"
    archivo_noiseprint = "TFG/huellas/huellasNoiseprint/iphone15/IMG_7887.JPG.npz"
    
    # 3. Construimos las rutas de salida apuntando a la nueva carpeta
    salida_prnu = os.path.join(carpeta_destino, "visualizacion_PRNU.png")
    salida_noiseprint = os.path.join(carpeta_destino, "visualizacion_NOISEPRINT.png")
    
    # 4. Ejecución de las pruebas de existencia y generación
    if os.path.exists(archivo_prnu):
        visualizar_prnu(archivo_prnu, salida_prnu)
    else:
        print(f"No se encuentra el archivo PRNU en: {archivo_prnu}. Revisa la ruta.")
        
    if os.path.exists(archivo_noiseprint):
        visualizar_noiseprint(archivo_noiseprint, salida_noiseprint)
    else:
        print(f"No se encuentra el archivo Noiseprint en: {archivo_noiseprint}. Revisa la ruta.")