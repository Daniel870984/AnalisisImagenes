import os
import warnings



# Silenciar advertencias de Python (Deprecated, FutureWarnings, etc.)
warnings.filterwarnings("ignore")

# Silenciar los mensajes internos de TensorFlow 0 = todo, 1 = info, 2 = warnings, 3 = errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Silenciar el logger de TensorFlow (Python)
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
    except:
        pass




import glob
import numpy as np
import time
import cv2

from PIL import Image, ExifTags, ImageOps
import prnu
# Funciones de PRNU
from prnu.functions import extract_single, zero_mean_total, wiener_dft, crosscorr_2d, pce

# Funciones del proyecto Noiseprint
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f, jpeg_qtableinv

# Librerías para métricas y visualización
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef


# ==============================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==============================
tamañoRecorte = 1024          # Tamaño del recorte central (1024x1024 píxeles)
carpetaDatos = "TFG/dataset"   # Carpeta con las fotos originales (organizadas por móvil)
carpetaHuellasNoiseprint = "TFG/huellas/huellasNoiseprint" # Carpeta para guardar los .npz procesados
carpetaMaestrasNoiseprint = "TFG/maestras/maestrasNoiseprint" # Carpeta para guardar las Huellas Maestras (.npy)
carpetaHuellasPRNU = "TFG/huellas/huellasPRNU" # Carpeta para guardar las huellas PRNU (.npy)
carpetaMaestrasPRNU = "TFG/maestras/maestrasPRNU" # Carpeta para guardar las Huellas Maestras PRNU (.npy)

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
        # Construimos la ruta completa (ej: "dataset/iphone15")
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
            
           # Eliminamos cualquier extensión para crear un nombre limpio (ej: 20221122_120811)
            nombreSinExt = os.path.splitext(nombreFoto)[0]
            
            # Definimos el nombre "existente", porque al principio los guardé con este formato
            archivoSalidaExistente = os.path.join(carpetaDestino, nombreFoto + ".npz")
            
            # Definimos el nombre sin extension -> 20221122_120811.npz
            archivoSalidaLimpio = os.path.join(carpetaDestino, nombreSinExt + ".npz")

            # Comprobamos si existe cualquiera de las dos versiones
            if os.path.exists(archivoSalidaExistente) or os.path.exists(archivoSalidaLimpio):
                print(f"   [SKIP] Ya existe la huella para: {nombreFoto}")
                continue

            try:
                # Leer imagen y calidad
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
            
            # Nombres de archivo, en PRNU es npy, no npz, porque solo guardamos la matriz de ruido sin el QF ni otros datos.
            nombreSinExt = os.path.splitext(nombreFoto)[0]
            archivoSalidaExistente = os.path.join(carpetaDestino, nombreFoto + ".npy")
            archivoSalidaLimpio = os.path.join(carpetaDestino, nombreSinExt + ".npy")

            if os.path.exists(archivoSalidaExistente) or os.path.exists(archivoSalidaLimpio):
                print(f"   [SKIP] Ya existe la huella PRNU para: {nombreFoto}")
                continue

            try:
                # Leer imagen
                img = np.asarray(Image.open(foto_path))
                
                # Extraer PRNU: Pasa la imagen por el filtro Wavelet
                res = extract_single(img)

                # Recortar centro
                h, w = res.shape
                if h < tamañoRecorte or w < tamañoRecorte:
                    print(f"   [AVISO] {nombreFoto} muy pequeña ({h}x{w}). Ignorada.")
                    continue

                cy, cx = h // 2, w // 2
                dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
                recorte = res[cy-dy:cy+dy, cx-dx:cx+dx]

                # Guardar npy
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
        # Construimos la ruta completa (ej: "huellasNoiseprint/iphone15")
        rutaCompleta = os.path.join(carpetaHuellasNoiseprint, d)
    
        # Comprobamos si es una carpeta (modelo) y lo añadimos a la lista de modelos, si no, se ignora.
        if os.path.isdir(rutaCompleta):
            modelos.append(d) 
    
    if not os.path.exists(carpetaMaestrasNoiseprint):
        os.makedirs(carpetaMaestrasNoiseprint)

    # Para cada modelo, calculamos la huella maestra (media de todas las huellas individuales)
    for modelo in modelos:
        rutaNPZ = os.path.join(carpetaHuellasNoiseprint, modelo, "*.npz")
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
            rutaSalida = os.path.join(carpetaMaestrasNoiseprint, f"MAESTRA_{modelo}.npy")
            np.save(rutaSalida, master)
            print(f"   [GUARDADO] {rutaSalida}")
        else:
            print("   No se pudo calcular la media.")



def entrenamientoPRNU():
    print("\n--- FASE 2: CÁLCULO DE HUELLAS MAESTRAS (PRNU) ---")
    
    if not os.path.exists(carpetaHuellasPRNU):
        print("Error: No hay carpeta de huellas PRNU. Ejecuta la Fase 1 primero.")
        return

    modelos = []
    todosLosItems = os.listdir(carpetaHuellasPRNU)

    for d in todosLosItems:
        # Construimos la ruta completa (ej: "huellasPRNU/iphone15")
        rutaCompleta = os.path.join(carpetaHuellasPRNU, d)
    
        # Comprobamos si es una carpeta (modelo) y lo añadimos a la lista de modelos, si no, se ignora.
        if os.path.isdir(rutaCompleta):
            modelos.append(d) 
    
    if not os.path.exists(carpetaMaestrasPRNU):
        os.makedirs(carpetaMaestrasPRNU)

    # Para cada modelo, calculamos la huella maestra (media de todas las huellas individuales)
    for modelo in modelos:
        rutaNPY = os.path.join(carpetaHuellasPRNU, modelo, "*.npy")
        archivos = glob.glob(rutaNPY)
        
        if not archivos:
            print(f"-> {modelo}: No hay archivos .npy.")
            continue

        print(f"-> Calculando promedio de '{modelo}' con {len(archivos)} huellas PRNU...")
        
        suma = None
        count = 0
        
        for archivo in archivos:
            try:
                # En PRNU, el .npy contiene directamente la matriz (no es un diccionario como el .npz)
                huella = np.load(archivo)
                if suma is None:
                    suma = huella.astype(np.float64) # Convertimos a float64 para evitar problemas de precisión
                else:
                    suma += huella
                count += 1
            except:
                print(f"   Error leyendo {os.path.basename(archivo)}")

        if count > 0:
            huella_media = suma / count # Calculamos la media para obtener la huella maestra del modelo
            
            # En PRNU, es necesario aplicar el zero-mean total y el filtro Wiener DFT para limpiar la huella maestra, ya que la media puede contener ruido no deseado
            print("   Aplicando Zero-Mean Total y filtro Wiener DFT...")
            huella_media_zm = zero_mean_total(huella_media) 
            master_limpia = wiener_dft(huella_media_zm, huella_media_zm.std(ddof=1))
            
            # Guardamos la huella maestra final ya limpia
            rutaSalida = os.path.join(carpetaMaestrasPRNU, f"MAESTRA_PRNU_{modelo}.npy")
            np.save(rutaSalida, master_limpia)
            print(f"   [GUARDADO] {rutaSalida}")
        else:
            print("   No se pudo calcular la media.")

# ================================
# 3. FASE DE VERIFICACIÓN / TEST
# ================================
def testNoiseprint():
    print("\n--- FASE 3: VERIFICAR UNA IMAGEN ---")
    
    # Buscar modelos disponibles (Maestras)
    huellasMaestras = glob.glob(os.path.join(carpetaMaestrasNoiseprint, "MAESTRA_*.npy"))
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
            rutaMaster = os.path.join(carpetaMaestrasNoiseprint, f"MAESTRA_{modelo}.npy")
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

    
def testPRNU():
    print("\n--- FASE 3: VERIFICAR UNA IMAGEN (PRNU) ---")
    
    # Buscar modelos disponibles (Maestras PRNU)
    huellasMaestras = glob.glob(os.path.join(carpetaMaestrasPRNU, "MAESTRA_PRNU_*.npy"))
    if not huellasMaestras:
        print("Error: No hay huellas maestras PRNU. Ejecuta la Fase 2 primero.")
        return
    
    # Limpiamos el nombre para sacar solo el modelo
    modelosDisp = [os.path.basename(f).replace("MAESTRA_PRNU_", "").replace(".npy", "") for f in huellasMaestras]
    print(f"Modelos conocidos: {modelosDisp}")

    # Pedir ruta de la foto
    rutaImagen = input("Arrastra aquí la imagen a analizar y pulsa Enter: ").strip().strip('"').strip("'")
    
    if not os.path.exists(rutaImagen):
        print("Error: El archivo no existe.")
        return

    print(f"Analizando: {os.path.basename(rutaImagen)}...")
    
    try:
        # Extraer huella al vuelo (Motor PRNU)
        img = np.asarray(Image.open(rutaImagen))
        res = extract_single(img)
        
        # Recorta para comparar con las maestras (1024x1024 del centro)
        h, w = res.shape
        if h < tamañoRecorte or w < tamañoRecorte:
            print("Error: La imagen es demasiado pequeña para compararla.")
            return

        cy, cx = h // 2, w // 2
        dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
        huellaTest = res[cy-dy:cy+dy, cx-dx:cx+dx]

        # Comparar
        print(f"\n{'MODELO':<15} | {'CORRELACIÓN PCE (¡Mayor es mejor!)':<35}")
        print("-" * 55)
        
        mejorModelo = "Desconocido"
        maxPCE = float('-inf') # Iniciamos en menos infinito porque buscamos el MÁXIMO a diferencia de la distancia que buscaba el mínimo

        for modelo in modelosDisp:
            rutaMaster = os.path.join(carpetaMaestrasPRNU, f"MAESTRA_PRNU_{modelo}.npy")
            master = np.load(rutaMaster)
            
            # --- Métrica PRNU: Correlación Cruzada 2D y PCE ---
            cc = crosscorr_2d(master, huellaTest)
            pce_val = pce(cc)['pce']
            
            print(f"{modelo:<15} | {pce_val:.4f}")
            
            # Si el pico de correlación es el más alto hasta ahora, es nuestro candidato
            if pce_val > maxPCE:
                maxPCE = pce_val
                mejorModelo = modelo
        
        print("-" * 55)
        print(f" CONCLUSIÓN: La imagen pertenece al -> [ {mejorModelo.upper()} ]\n")

    except Exception as e:
        print(f"Error durante el análisis: {e}")


# ====================================
# FUNCIÓN PARA GENERAR MÉTRICAS Y PDF 
# ====================================
def evaluar_y_generar_pdf(y_real, y_pred, clases_unicas, nombre_metodo, valores_metrica=None, nombre_valor=""):    
    print(f"\n" + "="*50)
    print(f"RESULTADOS MÉTRICAS: {nombre_metodo.upper()}")
    print("="*50)

    # Calculamos las métricas
    reporte = classification_report(y_real, y_pred, labels=clases_unicas, zero_division=0)
    acc = accuracy_score(y_real, y_pred)
    mcc = matthews_corrcoef(y_real, y_pred)
    cm = confusion_matrix(y_real, y_pred, labels=clases_unicas)
    
    # Imprimir en consola
    print("\n--- CLASSIFICATION REPORT ---")
    print(reporte)
    print(f"Accuracy Global: {acc:.4f} ({(acc*100):.2f}%)")
    print(f"Matthews Corr. Coef. (MCC): {mcc:.4f}")

    # Calculamos y mostramos la media de la métrica (PCE o Distancia)
    texto_extra = ""
    if valores_metrica and len(valores_metrica) > 0:
        # Evitamos valores infinitos en la media por si falló alguna foto
        valores_limpios = [v for v in valores_metrica if v != float('inf') and v != float('-inf')]
        if valores_limpios:
            media = sum(valores_limpios) / len(valores_limpios)
            texto_extra = f"Media de la fuerza de señal ({nombre_valor}): {media:.4f}\n"
            print(texto_extra.strip())

    # Imprimir falsos positivos y negativos por terminal
    for i, clase in enumerate(clases_unicas):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP 
        FN = cm[i, :].sum() - TP
        linea_fp_fn = f"{clase:<22} -> FP: {FP:<4} | FN: {FN:<4}"
        print(linea_fp_fn)

    # --- CONFIGURACIÓN DE LA GRÁFICA (Ajustada para el PDF limpio) ---
    # Reducimos el tamaño de la figura (ej: 10x8) ya que ya no lleva el texto gigante abajo
    plt.figure(figsize=(10, 8)) 
    

    # 1. Diccionario para mapear los nombres tal y como te vienen a como quiere Ricardo
    traduccion_nombres = {
        'iphone14': 'iPhone 14',
        'iphone14_2': 'iPhone 14',
        'iphone15': 'iPhone 15',
        'samsungS21': 'S. S21',
        'samsungNote10': 'S. Note 10',
        'samsungNote10_flatfield': 'S. Note 10\n(Flatfield)',
        'iphone14_personal': 'iPhone 14'
    }
    
    # 2. Convertimos la lista de clases usando el diccionario (si no está, se queda igual)
    clases_formateadas = [traduccion_nombres.get(clase, clase) for clase in clases_unicas]
    # Pintamos el Heatmap ocupando bien el espacio
    # annot_kws={'size': 16} es el truco para que los números de las celdas se vean grandes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases_formateadas, yticklabels=clases_formateadas,
                annot_kws={'size': 35}) # <-- FUENTE DE LAS CELDAS MAS GRANDE
    
    # Ajustamos también el tamaño de los títulos y etiquetas para que vayan acorde
    plt.title(f'MC - {nombre_metodo}', fontsize=30, pad=20)
    plt.ylabel('Clase real', fontsize=16)
    plt.xlabel('Clase predicha', fontsize=16)
    
    # Agrandamos la fuente de las etiquetas de las clases en los ejes
    plt.xticks(rotation=30, ha='center', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    
    # Aseguramos la ruta y guardamos el PDF ajustado
    carpeta_destino = os.path.join("TFG", "estadisticas")
    os.makedirs(carpeta_destino, exist_ok=True)
    
    nombre_archivo = f"Estadísticas_{nombre_metodo}.pdf"
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    
    # bbox_inches='tight' se encargará de ajustar los márgenes para que no se corten los textos de los ejes
    plt.savefig(ruta_completa, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Gráfica guardada exitosamente (Métricas solo por terminal) en: {ruta_completa}")
    print("="*50)

def evaluacionGlobal():
    print("\n--- FASE 4: EVALUACIÓN MASIVA CON MATRIZ DE CONFUSIÓN ---")

    print("¿Qué dataset de prueba quieres evaluar?")
    print("1. Fotos Originales")
    print("2. Fotos de WhatsApp")
    print("3. Experimento Device ID")
    print("4. Experimento Filtro Lineal")
    print("5. Experimento Filtrado Belleza")
    print("6. Experimento Instagram In-App")
    opc_test = input("Elige una opción (1,2,3,4,5 o 6): ")
    
    if opc_test == '1':
        carpetaTests = "TFG/tests/test"
        etiqueta = "ORIGINALES"
        dirMaestrasNP = carpetaMaestrasNoiseprint 
        dirMaestrasPRNU = carpetaMaestrasPRNU     
    elif opc_test == '2':
        carpetaTests = "TFG/tests/testWhatsApp"
        etiqueta = "WHATSAPP"
        dirMaestrasNP = "TFG/maestras/maestrasNoiseprintWhatsApp"
        dirMaestrasPRNU = "TFG/maestras/maestrasPRNUWhatsApp"
    elif opc_test == '3':
        carpetaTests = "TFG/tests/testDevice"
        etiqueta = "DEVICE"
        dirMaestrasNP = "TFG/maestras/maestrasNoiseprintDevice"
        dirMaestrasPRNU = "TFG/maestras/maestrasPRNUDevice"     
    elif opc_test == '4':
        carpetaTests = "TFG/tests/testFiltroLinealSevero"
        etiqueta = "FILTRO LINEAL SEVERO"
        dirMaestrasNP = "TFG/maestras/maestrasNoiseprintFiltroLineal" 
        dirMaestrasPRNU = "TFG/maestras/maestrasPRNUFiltroLineal"
    elif opc_test == '5':
        carpetaTests = "TFG/tests/testBelleza"
        etiqueta = "FILTRO BELLEZA"
        dirMaestrasNP = carpetaMaestrasNoiseprint 
        dirMaestrasPRNU = carpetaMaestrasPRNU
    elif opc_test == '6':
        carpetaTests = "TFG/tests/testInstagram"
        etiqueta = "INSTAGRAM"
        dirMaestrasNP = "TFG/maestras/maestrasNoiseprintInstagram"
        dirMaestrasPRNU = "TFG/maestras/maestrasPRNUInstagram"
    else:
        print("Opción no válida. Cancelando evaluación.")
        return
    
    if not os.path.exists(carpetaTests):
        print(f"Error: No existe la carpeta '{carpetaTests}'.")
        return
    
    carpeta_cache_global = os.path.join("TFG", "huellasTests", etiqueta)

    # Cargar maestras
    maestras_np = {}
    for ruta in glob.glob(os.path.join(dirMaestrasNP, "MAESTRA_*.npy")):
        if "PRNU" not in ruta: 
            modelo = os.path.basename(ruta).replace("MAESTRA_", "").replace(".npy", "")
            maestras_np[modelo] = np.load(ruta)

    maestras_prnu = {}
    for ruta in glob.glob(os.path.join(dirMaestrasPRNU, "MAESTRA_PRNU_*.npy")):
        modelo = os.path.basename(ruta).replace("MAESTRA_PRNU_", "").replace(".npy", "")
        maestras_prnu[modelo] = np.load(ruta)

    modelos_test = [d for d in os.listdir(carpetaTests) if os.path.isdir(os.path.join(carpetaTests, d))]
    
    modelos_pred_np = list(maestras_np.keys())
    modelos_pred_prnu = list(maestras_prnu.keys())
    
    # Listas para almacenar resultados reales y predichos para ambas técnicas, así como las distancias y PCE para análisis adicional
    lista_reales = []
    lista_pred_np = []
    lista_pred_prnu = []
    lista_distancias_np = [] 
    lista_pce_prnu = []

    tiempo_inicio = time.time()

    # EVALUACIÓN FOTO A FOTO
    for modelo_real in modelos_test:

        ruta_jpg = os.path.join(carpetaTests, modelo_real, "*.jpg")
        ruta_jpeg = os.path.join(carpetaTests, modelo_real, "*.jpeg")
        listaFotos = glob.glob(ruta_jpg) + glob.glob(ruta_jpeg)
        
        total_fotos = len(listaFotos)
        if total_fotos == 0: continue
            
        print(f"\n-> Evaluando {total_fotos} fotos del modelo real: [ {modelo_real.upper()} ]")

        # Carpetas de caché específicas por método
        cache_modelo_np = os.path.join(carpeta_cache_global, "Noiseprint", modelo_real)
        cache_modelo_prnu = os.path.join(carpeta_cache_global, "PRNU", modelo_real)
        os.makedirs(cache_modelo_np, exist_ok=True)
        os.makedirs(cache_modelo_prnu, exist_ok=True)

        for i, foto_path in enumerate(listaFotos):
            nombreFoto = os.path.basename(foto_path)
            print(f"   Analizando {i+1}/{total_fotos}: {nombreFoto}...", end=" ")

            nombreSinExt = os.path.splitext(nombreFoto)[0]
            # Formatos e hilos de guardado según tus funciones de extracción originales:
            ruta_cache_np = os.path.join(cache_modelo_np, f"{nombreSinExt}.npz") # <-- NOISEPRINT USA .npz
            ruta_cache_prnu = os.path.join(cache_modelo_prnu, f"{nombreSinExt}.npy") # <-- PRNU USA .npy
            
            try:
                if os.path.exists(ruta_cache_np):
                    datos_npz = np.load(ruta_cache_np)
                    huella_test_np = datos_npz['noiseprint']
                else:
                # Extraemos ambas huellas al vuelo (Noiseprint y PRNU) para esta foto de prueba
                    img_np, _ = imread2f(foto_path, channel=1)
                    img_prnu = np.asarray(Image.open(foto_path))
                    try: QF = jpeg_qtableinv(foto_path)
                    except: QF = 200

                    res_np = genNoiseprint(img_np, QF)
                    h, w = res_np.shape
                    cy, cx = h // 2, w // 2
                    dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
                    huella_test_np = res_np[cy-dy:cy+dy, cx-dx:cx+dx]
                    np.savez(ruta_cache_np, noiseprint=huella_test_np, QF=QF)


                if os.path.exists(ruta_cache_prnu):
                    huella_test_prnu = np.load(ruta_cache_prnu)
                else:
                    img_prnu = np.asarray(Image.open(foto_path))
                    res_prnu = extract_single(img_prnu)
                    
                    h, w = res_prnu.shape
                    cy, cx = h // 2, w // 2
                    dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
                    huella_test_prnu = res_prnu[cy-dy:cy+dy, cx-dx:cx+dx]
                    
                    # Guardamos respetando tu archivo npy plano original
                    np.save(ruta_cache_prnu, huella_test_prnu)

                # --- COMPARACIÓN NOISEPRINT ---
                mejor_np = "Desconocido"
                menor_dist = float('inf')
                for mod_maestra, master_np in maestras_np.items():
                    dist = np.linalg.norm(huella_test_np - master_np)
                    if dist < menor_dist:
                        menor_dist = dist
                        mejor_np = mod_maestra

                # --- COMPARACIÓN PRNU ---
                mejor_prnu = "Desconocido"
                mayor_pce = float('-inf')
                for mod_maestra, master_prnu in maestras_prnu.items():
                    cc = crosscorr_2d(master_prnu, huella_test_prnu)
                    pce_val = pce(cc)['pce']
                    if pce_val > mayor_pce:
                        mayor_pce = pce_val
                        mejor_prnu = mod_maestra

                # Guardamos los resultados para las métricas globales
                lista_reales.append(modelo_real)
                lista_pred_np.append(mejor_np)
                lista_distancias_np.append(menor_dist)
                lista_pred_prnu.append(mejor_prnu)
                lista_pce_prnu.append(mayor_pce)

                
                print(f"NP: {mejor_np} | PRNU: {mejor_prnu}")

            except Exception as e:
                print(f" ERROR ({e})")
                continue
                #lista_reales.append(modelo_real)
                #lista_pred_np.append("Desconocido")
                #lista_pred_prnu.append("Desconocido")

    # MÉTRICAS Y PDF
    evaluar_y_generar_pdf(lista_reales, lista_pred_np, modelos_pred_np, f"NOISEPRINT_{etiqueta}", lista_distancias_np, "Distancia Euclidiana")
    evaluar_y_generar_pdf(lista_reales, lista_pred_prnu, modelos_pred_prnu, f"PRNU_{etiqueta}", lista_pce_prnu, "PCE")

    print("\n" + "-" * 60)
    print(f"Tiempo total de evaluación: {(time.time() - tiempo_inicio)/60:.1f} minutos.")


def crearDatasetFiltroLineal():
    """
    Aplica una transformación lineal puntual (Brillo y Contraste)
    a todas las imágenes de la carpeta dataset_test y las guarda
    en una nueva carpeta manteniendo la estructura de directorios.
    Ecuación matemática aplicada: Y = alpha * X + beta
    """
    carpeta_origen = "TFG/tests/test"
    carpeta_destino = "TFG/tests/testFiltroLinealSevero"
    
    # Parámetros de la transformación lineal
    alpha = 1.4 # Factor de contraste (>1 aumenta, <1 disminuye)
    beta = 80 # Factor de brillo (valores positivos aclaran, negativos oscurecen)
    
    print("\n" + "="*50)
    print(" INICIANDO APLICACIÓN DE FILTRO LINEAL (BRILLO Y CONTRASTE)")
    print(f" Parámetros: Contraste (alpha)={alpha} | Brillo (beta)={beta}")
    print("="*50)
    
    if not os.path.exists(carpeta_origen):
        print(f"[ERROR] No se encuentra la carpeta origen: {carpeta_origen}")
        return

    contador = 0
    for directorio_raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                ruta_origen = os.path.join(directorio_raiz, archivo)
                ruta_relativa = os.path.relpath(directorio_raiz, carpeta_origen)
                ruta_destino_dir = os.path.join(carpeta_destino, ruta_relativa)
                
                if not os.path.exists(ruta_destino_dir):
                    os.makedirs(ruta_destino_dir)
                    
                ruta_destino = os.path.join(ruta_destino_dir, archivo)
                
                # Solo procesar si la imagen no existe ya en el destino
                if not os.path.exists(ruta_destino):
                    try:
                        imagen = Image.open(ruta_origen)
                        
                        # Suma un valor constante (ej. +10) a cada píxel de la imagen
                        #imagen_brillo = Image.eval(imagen, lambda x: x + 50)
                        imagen_filtrada = Image.eval(
                            imagen, 
                            lambda x: max(0, min(255, int(alpha * (x - 128) + 128 + beta)))
                        )                        
                        imagen_filtrada.save(ruta_destino, quality=100)
                        contador += 1
                        print(f"[OK] Brillo modificado: {archivo}")
                    except Exception as e:
                        print(f"[ERROR] Fallo en {archivo}: {e}")
                
    print("\n" + "-"*50)
    print(f" Proceso finalizado. {contador} imágenes nuevas generadas.")
    print(f" Carpeta destino: {carpeta_destino}")
    print("-"*50 + "\n")


def crearDatasetFiltroBelleza():
   
    carpeta_origen = "TFG/tests/test"
    carpeta_destino = "TFG/tests/testBelleza"
    
    print("\n" + "="*50)
    print(" INICIANDO APLICACIÓN DE FILTRO NO LINEAL (BELLEZA)")
    print("="*50)
    
    if not os.path.exists(carpeta_origen):
        print(f"[ERROR] No se encuentra la carpeta origen: {carpeta_origen}")
        return

    contador = 0
    for directorio_raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                ruta_origen = os.path.join(directorio_raiz, archivo)
                ruta_relativa = os.path.relpath(directorio_raiz, carpeta_origen)
                ruta_destino_dir = os.path.join(carpeta_destino, ruta_relativa)
                
                if not os.path.exists(ruta_destino_dir):
                    os.makedirs(ruta_destino_dir)
                    
                ruta_destino = os.path.join(ruta_destino_dir, archivo)
                
                if not os.path.exists(ruta_destino):
                    try:
                        # Leer imagen con OpenCV
                        imagen = cv2.imread(ruta_origen)
                        
                        # Aplicar Filtro Bilateral (NO LINEAL)
                        # Parámetros: d=15 (diámetro), sigmaColor=75, sigmaSpace=75
                        # Esto "plancha" las texturas pero mantiene los bordes
                        suavizada = cv2.bilateralFilter(imagen, 15, 75, 75)
                        
                        # Aplicar CLAHE (Contraste local NO LINEAL)
                        # Convertimos a formato LAB para alterar solo la luminosidad, no los colores
                        lab = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        
                        limg = cv2.merge((cl, a, b))
                        imagen_final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                        
                        # Guardar imagen procesada con máxima calidad
                        cv2.imwrite(ruta_destino, imagen_final, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        
                        contador += 1
                        print(f"[OK] Procesada: {archivo}")
                    except Exception as e:
                        print(f"[ERROR] Fallo en {archivo}: {e}")
                
    print("\n" + "-"*50)
    print(f" Proceso finalizado. {contador} imágenes nuevas generadas.")
    print(f" Carpeta destino: {carpeta_destino}")
    print("-"*50 + "\n")


def auditar_dataset(carpeta_fotos):
    print(f"\n" + "="*70)
    print(f"INICIANDO AUDITORÍA FORENSE DE SENSORES EN: {carpeta_fotos}")
    print("="*70)
    
    # Contadores para el resumen final
    fotos_validas = 0
    fotos_sospechosas = 0

    # Revisamos que la carpeta exista
    if not os.path.exists(carpeta_fotos):
        print(f"❌ ERROR: La carpeta '{carpeta_fotos}' no existe.")
        return

    for archivo in os.listdir(carpeta_fotos):
        if archivo.lower().endswith(('.jpg', '.jpeg')):
            ruta = os.path.join(carpeta_fotos, archivo)
            
            try:
                img = Image.open(ruta)
                # Extraemos los datos EXIF brutos
                exif_bruto = img._getexif()
                
                if exif_bruto is None:
                    print(f"❌ {archivo:<15} -> [ALERTA] Sin EXIF. (Posible captura o compresión)")
                    fotos_sospechosas += 1
                    continue

                # Mapeamos los EXIF a nombres legibles
                exif = {ExifTags.TAGS[k]: v for k, v in exif_bruto.items() if k in ExifTags.TAGS}
                
                # Extraemos las etiquetas clave
                modelo = exif.get('Model', 'Desconocido')
                lente = exif.get('LensModel', 'Desconocido')
                focal_35mm = exif.get('FocalLengthIn35mmFilm', 'N/A')
                
                # REGLAS DE ORO PARA EL iPHONE 14 PRO (Cámara Principal 1x)
                # 1. El modelo debe ser de la familia iPhone 14
                es_iphone14 = "iPhone 14" in str(modelo)
                
                # 2. La longitud focal equivalente a 35mm debe ser 24mm (el estándar del 1x)
                es_focal_correcta = str(focal_35mm) == "24"
                
                # Evaluamos
                if es_iphone14 and es_focal_correcta:
                    print(f" {archivo:<15} -> OK ({modelo} | 1x: {focal_35mm}mm | Lente: {lente})")
                    fotos_validas += 1
                else:
                    print(f" {archivo:<15} -> [DESCARTAR] Focal incorrecta: {focal_35mm}mm (Lente: {lente})")
                    fotos_sospechosas += 1
                    
            except Exception as e:
                print(f" {archivo:<15} -> ERROR DE LECTURA: {e}")
                fotos_sospechosas += 1

    print("\n" + "="*70)
    print(" RESUMEN DE LA AUDITORÍA")
    print(f"✅ Fotos válidas (Cámara Principal 1x): {fotos_validas}")
    print(f"⚠️ Fotos para descartar (0.5x, Selfie o Sin EXIF): {fotos_sospechosas}")
    
    if fotos_sospechosas > 0:
        print("\n RECOMENDACIÓN: Elimina las fotos sospechosas antes de generar la Huella Maestra.")
    else:
        print("\n DATASET PURO: Listo para generar la Huella Maestra (PRNU/Noiseprint).")
    print("="*70 + "\n")




# ===============
# MENÚ PRINCIPAL
# ===============
def main():
    while True:
        print("\n" + "="*48)
        print("  CLASIFICADOR FORENSE (NOISEPRINT vs PRNU)")
        print("================================================")
        print("--- NOISEPRINT (Deep Learning) ---")
        print("1. Extraer huellas Noiseprint")
        print("2. Calcular Huella Maestra Noiseprint")
        print("3. Verificar imagen con Noiseprint")
        print("")
        print("--- PRNU (Sensor Físico) ---")
        print("4. Extraer huellas PRNU")
        print("5. Calcular Huella Maestra PRNU")
        print("6. Verificar imagen con PRNU")
        print("")
        print("7. Evaluación Global + Estadísticas")
        print("")
        print("8. Crear Dataset con Filtro Lineal")
        print("")
        print("9. Conversion con Filtro de Belleza")
        print("")
        print("10. Auditar dataset")
        print("")
        print("11. Probar generación de TEST de PDF")
        print("")

        print("12. Salir")
        print("================================================")
        
        opcion = input("\nElige una opción (1-12): ")

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
            evaluacionGlobal()
            break
        elif opcion == '8':
            crearDatasetFiltroLineal()
            break
        elif opcion == '9':
            crearDatasetFiltroBelleza()
            break
        elif opcion == '10':
            auditar_dataset("TFG/dataset/iphone14_personal")
        elif opcion == '11':
            # Para probar la generación de PDF
            print("\nGenerando PDF de prueba...")
            clases = ["iphone14", "iphone15", "samsungS21", "samsungNote10","samsungNote10_flatfield"]
            reales =       ["iphone14", "iphone14", "iphone15", "samsungS21", "samsungS21", "samsungNote10", "samsungNote10_flatfield"]
            predicciones = ["iphone14", "samsungNote10", "iphone15", "iphone15", "samsungS21", "samsungNote10", "samsungNote10_flatfield"]
            evaluar_y_generar_pdf(reales, predicciones, clases, "PRUEBA_RAPIDA")
        elif opcion == '12':
            print("¡Bye!")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()