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

# --- NUEVAS LIBRERÍAS PARA ESTADÍSTICAS Y PDF ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef


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
            huella_media = suma / count # Calculamos la media (Estimador de Máxima Verosimilitud)
            
            # --- PASO CRÍTICO DE PRNU: LIMPIEZA DE ARTEFACTOS ---
            # Borramos patrones compartidos por todos los móviles de esa marca
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
        # 1. Extraer huella al vuelo (Motor PRNU)
        img = np.asarray(Image.open(rutaImagen))
        res = extract_single(img)
        
        # 2. Recorta para comparar con las maestras (1024x1024 del centro)
        h, w = res.shape
        if h < tamañoRecorte or w < tamañoRecorte:
            print("Error: La imagen es demasiado pequeña para compararla.")
            return

        cy, cx = h // 2, w // 2
        dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
        huellaTest = res[cy-dy:cy+dy, cx-dx:cx+dx]

        # 3. Comparar
        print(f"\n{'MODELO':<15} | {'CORRELACIÓN PCE (¡Mayor es mejor!)':<35}")
        print("-" * 55)
        
        mejorModelo = "Desconocido"
        maxPCE = float('-inf') # ATENCIÓN: Iniciamos en menos infinito porque buscamos el MÁXIMO

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
def evaluar_y_generar_pdf(y_real, y_pred, clases_unicas, nombre_metodo):
    print(f"\n" + "="*50)
    print(f"📊 RESULTADOS MÉTRICAS: {nombre_metodo.upper()} 📊")
    print("="*50)

    reporte = classification_report(y_real, y_pred, labels=clases_unicas, zero_division=0)
    print("\n--- CLASSIFICATION REPORT ---")
    print(reporte)

    acc = accuracy_score(y_real, y_pred)
    mcc = matthews_corrcoef(y_real, y_pred)
    print(f"Accuracy Global: {acc:.4f} ({(acc*100):.2f}%)")
    print(f"Matthews Corr. Coef. (MCC): {mcc:.4f}")

    cm = confusion_matrix(y_real, y_pred, labels=clases_unicas)
    
    print("\n--- FALSOS POSITIVOS (FP) Y FALSOS NEGATIVOS (FN) ---")
    for i, clase in enumerate(clases_unicas):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP 
        FN = cm[i, :].sum() - TP
        print(f"{clase:<22} -> FP: {FP:<4} | FN: {FN:<4}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases_unicas, yticklabels=clases_unicas)
    
    plt.title(f'Matriz de Confusión - {nombre_metodo}')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    nombre_archivo = f"Estadísticas_{nombre_metodo}.pdf"
    plt.savefig(nombre_archivo, format='pdf', bbox_inches='tight')
    plt.close() 
    
    print(f"\n Gráfica guardada exitosamente como: {nombre_archivo}")
    print("="*50)



def evaluacionGlobal():
    print("\n--- FASE 4: EVALUACIÓN MASIVA CON MATRIZ DE CONFUSIÓN ---")

    print("¿Qué dataset de prueba quieres evaluar?")
    print("1. Fotos Originales (TFG/test)")
    print("2. Fotos de WhatsApp (TFG/testWhatsApp)")
    opc_test = input("Elige una opción (1 o 2): ")
    
    if opc_test == '1':
        carpetaTests = "TFG/test"
        etiqueta = "ORIGINALES"
    elif opc_test == '2':
        carpetaTests = "TFG/testWhatsApp"
        etiqueta = "WHATSAPP"
    else:
        print("Opción no válida. Cancelando evaluación.")
        return
    
    if not os.path.exists(carpetaTests):
        print(f"Error: No existe la carpeta '{carpetaTests}'.")
        return

    # 1. PREPARACIÓN: Cargar maestras
    maestras_np = {}
    for ruta in glob.glob(os.path.join(carpetaMaestrasNoiseprint, "MAESTRA_*.npy")):
        if "PRNU" not in ruta: 
            modelo = os.path.basename(ruta).replace("MAESTRA_", "").replace(".npy", "")
            maestras_np[modelo] = np.load(ruta)

    maestras_prnu = {}
    for ruta in glob.glob(os.path.join(carpetaMaestrasPRNU, "MAESTRA_PRNU_*.npy")):
        modelo = os.path.basename(ruta).replace("MAESTRA_PRNU_", "").replace(".npy", "")
        maestras_prnu[modelo] = np.load(ruta)

    modelos_test = [d for d in os.listdir(carpetaTests) if os.path.isdir(os.path.join(carpetaTests, d))]
    
    modelos_pred_np = list(maestras_np.keys()) + ["Desconocido"]
    modelos_pred_prnu = list(maestras_prnu.keys()) + ["Desconocido"]
    
    #Listas para almacenar resultados reales y predichos
    lista_reales = []
    lista_pred_np = []
    lista_pred_prnu = []

    tiempo_inicio = time.time()

    # 2. EVALUACIÓN FOTO A FOTO
    for modelo_real in modelos_test:
        # --- MODIFICADO: Buscamos ambas extensiones y sumamos los resultados ---
        ruta_jpg = os.path.join(carpetaTests, modelo_real, "*.jpg")
        ruta_jpeg = os.path.join(carpetaTests, modelo_real, "*.jpeg")
        listaFotos = glob.glob(ruta_jpg) + glob.glob(ruta_jpeg)
        # -----------------------------------------------------------------------
        
        total_fotos = len(listaFotos)
        if total_fotos == 0: continue
            
        print(f"\n-> Evaluando {total_fotos} fotos del modelo real: [ {modelo_real.upper()} ]")

        for i, foto_path in enumerate(listaFotos):
            nombreFoto = os.path.basename(foto_path)
            print(f"   Analizando {i+1}/{total_fotos}: {nombreFoto}...", end=" ")
            
            try:
                # --- EXTRACCIÓN AL VUELO ---
                img_np, _ = imread2f(foto_path, channel=1)
                img_prnu = np.asarray(Image.open(foto_path))
                try: QF = jpeg_qtableinv(foto_path)
                except: QF = 200

                res_np = genNoiseprint(img_np, QF)
                res_prnu = extract_single(img_prnu)

                h, w = res_np.shape
                cy, cx = h // 2, w // 2
                dy, dx = tamañoRecorte // 2, tamañoRecorte // 2
                
                huella_test_np = res_np[cy-dy:cy+dy, cx-dx:cx+dx]
                huella_test_prnu = res_prnu[cy-dy:cy+dy, cx-dx:cx+dx]

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
                lista_pred_prnu.append(mejor_prnu)
                
                print(f"NP: {mejor_np} | PRNU: {mejor_prnu}")

            except Exception as e:
                print(f" ERROR ({e})")
                lista_reales.append(modelo_real)
                lista_pred_np.append("Desconocido")
                lista_pred_prnu.append("Desconocido")

    # 3. MÉTRICAS Y PDF
    evaluar_y_generar_pdf(lista_reales, lista_pred_np, modelos_pred_np, f"NOISEPRINT_{etiqueta}")
    evaluar_y_generar_pdf(lista_reales, lista_pred_prnu, modelos_pred_prnu, f"PRNU_{etiqueta}")

    print("\n" + "-" * 60)
    print(f"Tiempo total de evaluación: {(time.time() - tiempo_inicio)/60:.1f} minutos.")

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
        print("7. Evaluación Global + Estadísticas")
        print("")
        print("8. Salir")
        print("9. [TEST RÁPIDO] Probar generación de PDF")
        print("================================================")
        
        opcion = input("\nElige una opción (1-8): ")

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
            print("¡Bye!")
            break
        elif opcion == '9':
            # --- PRUEBA RÁPIDA DE LA FUNCIÓN ---
            print("\nGenerando PDF de prueba...")
            clases = ["iphone14", "iphone15", "samsungS21", "Desconocido"]
            reales =       ["iphone14", "iphone14", "iphone15", "samsungS21", "samsungS21"]
            predicciones = ["iphone14", "Desconocido", "iphone15", "iphone15", "samsungS21"]
            evaluar_y_generar_pdf(reales, predicciones, clases, "PRUEBA_RAPIDA")
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()