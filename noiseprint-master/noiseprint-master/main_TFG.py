import os
import warnings



# 1. Silenciar advertencias de Python (Deprecated, FutureWarnings, etc.)
warnings.filterwarnings("ignore")

# 2. Silenciar los mensajes internos de TensorFlow (C++)
# 0 = todo, 1 = info, 2 = warnings, 3 = errors (solo fatal)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 3. Silenciar el logger de TensorFlow (Python)
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
    except:
        pass

# ------------------------------------------



import glob
import numpy as np
import time

from PIL import Image
import prnu
# --- Importamos funciones de PRNU ---
from prnu.functions import extract_single, zero_mean_total, wiener_dft, crosscorr_2d, pce

# --- Importamos las funciones del proyecto Noiseprint ---
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f, jpeg_qtableinv


# ==============================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==============================
tamañoRecorte = 1024          # Tamaño del recorte central (1024x1024 píxeles)
carpetaDatos = "TFG/dataset"   # Carpeta con las fotos originales (organizadas por móvil)
carpetaHuellasNoiseprint = "TFG/huellasNoiseprint" # Carpeta para guardar los .npz procesados
carpetaMaestrasNoiseprint = "TFG/maestrasNoiseprint" # Carpeta para guardar las Huellas Maestras (.npy)
carpetaHuellasPRNU = "TFG/huellasPRNU" # Carpeta para guardar las huellas PRNU (.npy)
carpetaMaestrasPRNU = "TFG/maestrasPRNU" # Carpeta para guardar las Huellas Maestras PRNU (.npy)

# =======================
# 1. FASE DE EXTRACCIÓN
# =======================
def extraccionNoiseprint():
    print("\n--- FASE 1: EXTRACCIÓN Y RECORTES ---")
    
    # Detectar carpetas de modelos automáticamente
    if not os.path.exists(carpetaDatos):
        print(f"Error: No existe la carpeta que contiene las imágenes base'{carpetaDatos}'.")
        return

    modelos = []
    todosLosItems = os.listdir(carpetaDatos)

    for d in todosLosItems:
        # Construimos la ruta completa (ej: "dataset/iphone")
        rutaCompleta = os.path.join(carpetaDatos, d)
    
        # Comprobamos si es una carpeta (modelo) y lo añadimos a la lista de modelos, si no, se ignora.
        if os.path.isdir(rutaCompleta):
            modelos.append(d) 

    if not modelos:
        print("No se han encontrado carpetas de modelos dentro de 'dataset/'.")
        return

    print(f"Modelos detectados: {modelos}")
    confirm = input("¿Quieres procesar todas las fotos de estos modelos? (s/n): ")
    if confirm.lower() != 's': return

    # Aseguramos que la carpeta de huellas exista
    if not os.path.exists(carpetaHuellasNoiseprint):
        os.makedirs(carpetaHuellasNoiseprint)

    # Contamos el tiempo para medir el rendimiento
    tiempoInicio = time.time()
    totalFotos = 0

    for modelo in modelos:
        rutaFotos = os.path.join(carpetaDatos, modelo, "*.jpg")
        listaFotos = glob.glob(rutaFotos)
        
        carpetaDestino = os.path.join(carpetaHuellasNoiseprint, modelo)
        # Aseguramos que la carpeta de destino (huellasNoiseprint/iphone15) exista
        if not os.path.exists(carpetaDestino):
            os.makedirs(carpetaDestino)

        print(f"\n-> Procesando {modelo} ({len(listaFotos)} fotos)...")

        for foto_path in listaFotos:
            nombreFoto = os.path.basename(foto_path)
            archivoSalida = os.path.join(carpetaDestino, nombreFoto.replace(".jpg", ".npz"))
            
           # 1. ESTO ELIMINA CUALQUIER EXTENSIÓN (.jpg, .JPG, .jpeg...)
            nombreSinExt = os.path.splitext(nombreFoto)[0]
            
            # 2. DEFINIMOS EL NOMBRE QUE YA TIENES (con la posible doble extensión)
            # Para que te reconozca los que ya creaste como IMG_XXXX.JPG.npz
            archivoSalidaExistente = os.path.join(carpetaDestino, nombreFoto + ".npz")
            
            # 3. DEFINIMOS EL NOMBRE "LIMPIO" (el que debería ser)
            archivoSalidaLimpio = os.path.join(carpetaDestino, nombreSinExt + ".npz")

            # Comprobamos si existe cualquiera de las dos versiones
            if os.path.exists(archivoSalidaExistente) or os.path.exists(archivoSalidaLimpio):
                print(f"   [SKIP] Ya existe la huella para: {nombreFoto}")
                continue

            try:
                # Leer imagen y calidad (igual que en el código original)
                img, mode = imread2f(foto_path, channel=1) # Leemos la imagen en escala de grises (channel=1)
                try: QF = jpeg_qtableinv(foto_path) # Intentamos detectar la calidad JPEG, si falla, asumimos la máxima calidad (200)
                except: QF = 200

                # Extraer Noiseprint
                # La función carga la red neuronal experta (ej: net_jpg75), pasa la imagen por las 17 capas y devuelve res (el residuo)
                res = genNoiseprint(img, QF)

                # Recortar centro
                h, w = res.shape # Obtenemos las dimensiones de la huella completa
                if h < tamañoRecorte or w < tamañoRecorte:
                    print(f"   [AVISO] {nombreFoto} muy pequeña ({h}x{w}). Ignorada.")
                    continue

                cy, cx = h // 2, w // 2 # Coordenadas del centro
                dy, dx = tamañoRecorte // 2, tamañoRecorte // 2 # Mitad del tamaño de recorte (512 para 1024x1024)
                recorte = res[cy-dy:cy+dy, cx-dx:cx+dx] # Recortamos el centro de la huella para obtener exactamente 1024x1024 píxeles

                # Guardamos, son unos datos matemáticos, no una imagen.
                np.savez(archivoSalida, noiseprint=recorte, QF=QF)
                print(f"   [OK] {nombreFoto} (QF={QF})")
                totalFotos += 1

            except Exception as e:
                print(f"   [ERROR] {nombreFoto}: {e}")

    print(f"\n--- Fin de extracción. {totalFotos} nuevas huellas generadas en {time.time()-tiempoInicio:.1f}s. ---")



def extraccionPRNU():
    print("\n--- FASE 1: EXTRACCIÓN Y RECORTES (PRNU) ---")
    
    # Detectar carpetas de modelos automáticamente
    if not os.path.exists(carpetaDatos):
        print(f"Error: No existe la carpeta que contiene las imágenes base '{carpetaDatos}'.")
        return

    modelos = []
    todosLosItems = os.listdir(carpetaDatos)

    for d in todosLosItems:
        rutaCompleta = os.path.join(carpetaDatos, d)
        if os.path.isdir(rutaCompleta):
            modelos.append(d) 

    if not modelos:
        print("No se han encontrado carpetas de modelos dentro de 'dataset/'.")
        return

    print(f"Modelos detectados: {modelos}")
    confirm = input("¿Quieres procesar todas las fotos de estos modelos con PRNU? (s/n): ")
    if confirm.lower() != 's': return

    
    if not os.path.exists(carpetaHuellasPRNU):
        os.makedirs(carpetaHuellasPRNU)

    tiempoInicio = time.time()
    totalFotos = 0

    for modelo in modelos:
        rutaFotos = os.path.join(carpetaDatos, modelo, "*.jpg")
        listaFotos = glob.glob(rutaFotos)
        
        carpetaDestino = os.path.join(carpetaHuellasPRNU, modelo)
        if not os.path.exists(carpetaDestino):
            os.makedirs(carpetaDestino)

        print(f"\n-> Procesando {modelo} con PRNU ({len(listaFotos)} fotos)...")

        for foto_path in listaFotos:
            nombreFoto = os.path.basename(foto_path)
            
            # Nombres de archivo (cambiamos .npz por .npy)
            nombreSinExt = os.path.splitext(nombreFoto)[0]
            archivoSalidaExistente = os.path.join(carpetaDestino, nombreFoto + ".npy")
            archivoSalidaLimpio = os.path.join(carpetaDestino, nombreSinExt + ".npy")

            if os.path.exists(archivoSalidaExistente) or os.path.exists(archivoSalidaLimpio):
                print(f"   [SKIP] Ya existe la huella PRNU para: {nombreFoto}")
                continue

            try:
                # 1. Leer imagen: la librería espera un array numpy RGB de tipo uint8 [cite: 73, 141-144]
                img = np.asarray(Image.open(foto_path))
                
                # 2. Extraer PRNU: Pasa la imagen por el filtro Wavelet [cite: 1, 9, 31]
                # NOTA: No hace falta el QF porque PRNU ignora la calidad JPEG, busca defectos físicos [cite: 36, 148]
                res = extract_single(img)

                # 3. Recortar centro (La misma lógica que en Noiseprint)
                h, w = res.shape
                if h < tamañoRecorte or w < tamañoRecorte:
                    print(f"   [AVISO] {nombreFoto} muy pequeña ({h}x{w}). Ignorada.")
                    continue

                cy, cx = h // 2, w // 2
                dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
                recorte = res[cy-dy:cy+dy, cx-dx:cx+dx]

                # 4. Guardar: como PRNU es solo una matriz de ruido, usamos np.save puro (.npy)
                np.save(archivoSalidaLimpio, recorte)
                
                print(f"   [OK] {nombreFoto}")
                totalFotos += 1

            except Exception as e:
                print(f"   [ERROR] {nombreFoto}: {e}")

    print(f"\n--- Fin de extracción PRNU. {totalFotos} nuevas huellas generadas en {time.time()-tiempoInicio:.1f}s. ---")

# ===========================================
# 2. FASE DE ENTRENAMIENTO (CALCULAR MAESTRA)
# ===========================================
def entrenamientoNoiseprint():
    print("\n--- FASE 2: CÁLCULO DE HUELLAS MAESTRAS ---")
    
    if not os.path.exists(carpetaHuellasNoiseprint):
        print("Error: No hay carpeta de huellas. Ejecuta la Fase 1 primero.")
        return

    modelos = []
    todosLosItems = os.listdir(carpetaHuellasNoiseprint)

    for d in todosLosItems:
        # Construimos la ruta completa (ej: "huellas/iphone")
        rutaCompleta = os.path.join(carpetaHuellasNoiseprint, d)
    
        # Comprobamos si es una carpeta (modelo) y lo añadimos a la lista de modelos, si no, se ignora.
        if os.path.isdir(rutaCompleta):
            modelos.append(d) 
    
    if not os.path.exists(carpetaMaestras):
        os.makedirs(carpetaMaestras)

    # Para cada modelo, calculamos la huella maestra (media de todas las huellas individuales)
    for modelo in modelos:
        rutaNPZ = os.path.join(carpetaHuellasNoiseprintNoiseprint, modelo, "*.npz")
        archivos = glob.glob(rutaNPZ)
        
        if not archivos:
            print(f"-> {modelo}: No hay archivos .npz.")
            continue

        print(f"-> Calculando promedio de '{modelo}' con {len(archivos)} huellas...")
        
        suma = None
        count = 0
        
        for archivo in archivos:
            try:
                datos = np.load(archivo)
                huella = datos['noiseprint']
                if suma is None:
                    suma = huella.astype(np.float64) # Convertimos a float64 para evitar problemas de precisión al sumar muchas huellas
                else:
                    suma += huella
                count += 1
            except:
                print(f"   Error leyendo {os.path.basename(archivo)}")

        if count > 0:
            master = suma / count # Calculamos la media para obtener la huella maestra del modelo
            rutaSalida = os.path.join(carpetaMaestras, f"MAESTRA_{modelo}.npy")
            np.save(rutaSalida, master)
            print(f"   [GUARDADO] {rutaSalida}")
        else:
            print("   No se pudo calcular la media.")

# ================================
# 3. FASE DE VERIFICACIÓN / TEST
# ================================
def testNoiseprint():
    print("\n--- FASE 3: VERIFICAR UNA IMAGEN ---")
    
    # Buscar modelos disponibles (Maestras)
    huellasMaestras = glob.glob(os.path.join(carpetaMaestras, "MAESTRA_*.npy"))
    if not huellasMaestras:
        print("Error: No hay huellas maestras. Ejecuta la Fase 2 primero.")
        return
    
    modelosDisp = [os.path.basename(f).replace("MAESTRA_", "").replace(".npy", "") for f in huellasMaestras]
    print(f"Modelos conocidos: {modelosDisp}")

    # Pedir ruta de la foto
    rutaImagen = input("Arrastra aquí la imagen a analizar y pulsa Enter: ").strip().strip('"').strip("'")
    
    if not os.path.exists(rutaImagen):
        print("Error: El archivo no existe.")
        return

    print(f"Analizando: {os.path.basename(rutaImagen)}...")
    
    try:
        # Extraer huella al vuelo
        img, _ = imread2f(rutaImagen, channel=1)
        try: QF = jpeg_qtableinv(rutaImagen)
        except: QF = 200
        
        res = genNoiseprint(img, QF)
        
        # Recorta para comparar con las maestras (1024x1024 del centro)
        h, w = res.shape
        if h < tamañoRecorte or w < tamañoRecorte:
            print("Error: La imagen es demasiado pequeña para compararla.")
            return

        cy, cx = h // 2, w // 2
        dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
        huellaTest = res[cy-dy:cy+dy, cx-dx:cx+dx]

        # Comparar
        print(f"\n{'MODELO':<15} | {'DISTANCIA (Menos es mejor)':<25}")
        print("-" * 45)
        
        mejorModelo = "Desconocido"
        menosDist = float('inf')

        for modelo in modelosDisp:
            rutaMaster = os.path.join(carpetaMaestras, f"MAESTRA_{modelo}.npy")
            master = np.load(rutaMaster)
            
            # Distancia Euclidiana
            dist = np.linalg.norm(huellaTest - master)
            
            print(f"{modelo:<15} | {dist:.4f}")
            
            # Si es el mejor resultado hasta ahora, lo guardamos para la conclusión final
            if dist < menosDist:
                menosDist = dist
                mejorModelo = modelo
        
        print("-" * 45)
        print(f" CONCLUSIÓN: La imagen pertenece al -> [ {mejorModelo.upper()} ]\n")

    except Exception as e:
        print(f"Error durante el análisis: {e}")

# ===============
# MENÚ PRINCIPAL
# ===============
def main():
    while True:
        print("\n" + "="*48)
        print("  CLASIFICADOR FORENSE (NOISEPRINT vs PRNU)")
        print("================================================")
        print("--- NOISEPRINT (Deep Learning) ---")
        print("1. Extraer huellas Noiseprint (.npz)")
        print("2. Calcular Huella Maestra Noiseprint")
        print("3. Verificar imagen con Noiseprint")
        print("")
        print("--- PRNU (Sensor Físico) ---")
        print("4. Extraer huellas PRNU (.npy)")
        print("5. Calcular Huella Maestra PRNU")
        print("6. Verificar imagen con PRNU")
        print("")
        print("7. Salir")
        print("================================================")
        
        opcion = input("\nElige una opción (1-7): ")

        if opcion == '1':
            extraccionNoiseprint()
        elif opcion == '2':
            entrenamientoNoiseprint()
        elif opcion == '3':
            testNoiseprint()
        elif opcion == '4':
            extraccionPRNU()
        elif opcion == '5':
            entrenamientoPRNU()
        elif opcion == '6':
            testPRNU()
        elif opcion == '7':
            print("¡Bye!")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()