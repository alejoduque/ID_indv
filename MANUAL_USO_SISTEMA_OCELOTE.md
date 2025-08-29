# 📖 Manual de Usuario - Sistema de Identificación de Ocelotes

## 🚀 Guía de Instalación y Uso

### Prerrequisitos del Sistema
- Python 3.7 o superior
- Sistema operativo: Windows, macOS, o Linux
- Espacio en disco: mínimo 2GB libres
- Memoria RAM: mínimo 4GB recomendados

### Instalación

#### Paso 1: Preparar el Entorno
```bash
# Navegar al directorio del proyecto
cd "ruta/a/tu/proyecto"

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

#### Paso 2: Instalar Dependencias
```bash
pip install opencv-python matplotlib numpy
```

## 🎯 Uso del Sistema

### Para Comparar Imágenes Estáticas

#### Archivo: `ocelot_pattern_comparison.py`
```bash
# Colocar las imágenes en el directorio del proyecto con nombres:
# - Ocelote_compare_1.jpg
# - Ocelote_compare_2.jpg

python ocelot_pattern_comparison.py
```

**Archivos generados:**
- `ocelot_pattern_analysis.png` - Análisis detallado de 6 paneles
- `ocelot_enhanced_comparison.png` - Comparación lado a lado
- `ocelot_feature_matches.png` - Coincidencias de características (si se encuentran)

### Para Comparar Videos (Recomendado)

#### Archivo: `ocelot_video_comparison.py`

1. **Editar rutas de videos** en el script:
```python
# Líneas 142-143, cambiar por las rutas de tus videos
video1_path = "ruta/a/tu/primer_video.AVI"
video2_path = "ruta/a/tu/segundo_video.mp4"
```

2. **Ejecutar análisis**:
```bash
python ocelot_video_comparison.py
```

**Archivos generados:**
- `ocelot_best_matches.png` - Mejores coincidencias entre fotogramas
- `ocelot_analysis_results.json` - Resultados detallados en formato JSON
- `frames/` - Carpeta con fotogramas extraídos organizados por video

## 📊 Interpretación de Resultados

### Niveles de Confianza

| Puntuación | Confianza | Interpretación |
|------------|-----------|----------------|
| 70-100% | **ALTA** | Muy probable mismo individuo |
| 40-70% | **MEDIA** | Posiblemente mismo individuo |
| 25-40% | **BAJA-MEDIA** | Incierto, revisar manualmente |
| 0-25% | **BAJA** | Probablemente individuos diferentes |

### Factores que Afectan la Precisión

#### ✅ **Factores Positivos**
- Videos de alta resolución
- Buena iluminación infrarroja
- Ocelote en posición lateral (vista del costado)
- Múltiples ángulos en el video
- Patrones de manchas claramente visibles

#### ❌ **Factores Negativos**
- Videos borrosos o de baja resolución
- Iluminación deficiente
- Ocelote de frente o desde atrás únicamente
- Movimiento muy rápido
- Obstrucciones (vegetación, objetos)

## 🔧 Personalización y Ajustes

### Modificar Parámetros del Análisis

En `ocelot_video_comparison.py`, puedes ajustar:

```python
# Línea 25: Número de fotogramas a extraer
frames1, paths1 = extract_frames_from_video(video1_path, num_frames=12)

# Línea 45: Configuración CLAHE (mejoramiento de contraste)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

# Línea 70: Número de características SIFT
sift = cv2.SIFT_create(nfeatures=500)

# Línea 102: Umbral de coincidencia (más estricto = menos coincidencias)
if m.distance < 0.65 * n.distance:
```

### Para Diferentes Especies

El sistema puede adaptarse para otros felinos con patrones únicos:
- **Jaguar**: Ajustar umbral de características (rosetas más grandes)
- **Margay**: Usar más fotogramas (animal más pequeño)
- **Tigrillo**: Configuración similar al ocelote

## 📁 Estructura de Archivos

```
proyecto/
├── ocelot_pattern_comparison.py      # Análisis de imágenes estáticas
├── ocelot_video_comparison.py        # Análisis de videos (recomendado)
├── venv/                             # Entorno virtual de Python
├── frames/                           # Fotogramas extraídos (auto-generado)
│   ├── video1_name/
│   └── video2_name/
├── ocelot_analysis_results.json      # Resultados del análisis
├── ocelot_best_matches.png           # Visualización de mejores coincidencias
└── README.md                         # Documentación del proyecto
```

## 🚨 Solución de Problemas

### Error: "Could not open video"
- Verificar ruta del archivo
- Comprobar formato soportado (AVI, MP4, MOV, MKV)
- Asegurar que el archivo no esté corrupto

### Error: "Insufficient feature matches"
- Revisar calidad del video
- Intentar con diferentes fotogramas
- Ajustar parámetros de extracción de características

### Resultados inconsistentes
- Usar videos en lugar de imágenes estáticas
- Asegurar buena iluminación infrarroja
- Verificar que los patrones sean claramente visibles

## 📞 Soporte y Contacto

Para preguntas técnicas o mejoras al sistema:
1. Revisar este manual primero
2. Verificar mensajes de error en la terminal
3. Documentar el problema con capturas de pantalla
4. Incluir archivos de video/imagen problemáticos (si es posible)

## 🔄 Actualizaciones y Mejoras

### Versión Actual: 1.0
- Análisis básico de imágenes y videos
- Extracción de características SIFT y ORB
- Visualización de resultados
- Análisis estadístico

### Próximas Mejoras Planificadas
- Base de datos de individuos identificados
- Interfaz gráfica de usuario
- Análisis batch para múltiples videos
- Exportación de reportes automáticos
- Integración con sistemas de cámaras trampa

---

**Creado**: Agosto 2025  
**Versión**: 1.0  
**Proyecto**: MANAKAI - Monitoreo de Fauna