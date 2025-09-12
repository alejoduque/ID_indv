# 🐆 Sistema Avanzado de Identificación Individual de Ocelotes - Proyecto MANAKAI

> Sistema automatizado mejorado para identificación de individuos de ocelote usando análisis especializado de patrones de rosetas y manchas en videos de cámaras trampa


Las vidas humanas y animales se intersectan, ya sea a través del contacto físico directo o al habitar el mismo espacio en momentos diferentes. Los académicos de las humanidades ambientales han comenzado a investigar estas relaciones a través del campo emergente de los estudios multiespecíficos, construyendo sobre décadas de trabajo en historia animal, estudios feministas y epistemologías indígenas. Los contribuyentes a este volumen consideran las relaciones humano-animales entrelazadas de un mundo multiespecífico complejo, donde animales domésticos, animales salvajes y personas se cruzan en el camino, creando naturalezas-culturas híbridas. La tecnología, argumentan, estructura cómo los animales y los humanos comparten espacios. Desde la ropa hasta los automóviles y las computadoras, la tecnología actúa como mediadora y conectora de vidas a través del tiempo y el espacio. Facilita formas de observar, medir, mover y matar, así como controlar, contener, conservar y cooperar con los animales. "Compartiendo Espacios" nos desafía a analizar cómo la tecnología configura las relaciones humanas con el mundo no humano, explorando a los animales no humanos como parientes, compañeros, alimento, transgresores, entretenimiento y herramientas.



## 📋 Descripción

Este sistema utiliza técnicas avanzadas de visión por computadora y algoritmos especializados en patrones de felinos para identificar individuos de ocelote (_Leopardus pardalis_). El sistema analiza patrones únicos de rosetas, manchas sólidas y características de piel comparando videos de cámaras trampa. La herramienta está diseñada para apoyar estudios de conservación y monitoreo poblacional no invasivo con análisis visuales comprensivos.

## 🎯 Características Principales

### 🆕 **Detección de Patrones Específicos de Ocelote**
- **Detección de rosetas**: Algoritmos especializados para identificar patrones circulares/elípticos característicos
- **Análisis de manchas sólidas**: Detección de patrones de manchas usando análisis de blob
- **Validación de patrones**: Verificación de contraste centro-borde para confirmar rosetas auténticas

### 📊 **Sistema de Análisis Visual Avanzado**
- **Mapas de calor de patrones**: Visualización de similaridad de rosetas y manchas
- **Líneas de perfil de patrones**: Análisis de tendencias de coincidencia a través de fotogramas
- **Matriz de comparación**: Heatmaps específicos para cada tipo de patrón
- **Indicadores de confianza**: Bordes codificados por color (verde=alta, naranja=moderada, rojo=baja)

### 🔬 **Algoritmos de Comparación Mejorados**
- **Análisis de videos**: Procesamiento de múltiples fotogramas para mayor precisión
- **Algoritmos tradicionales**: SIFT y ORB para extracción de características generales
- **Comparación espacial**: Matching basado en posición, tamaño y confianza de patrones
- **Puntuación ponderada**: Prioriza coincidencias de patrones específicos sobre características generales

### 📋 **Interfaz y Documentación**
- **Línea de comandos flexible**: Acepta rutas de video como parámetros
- **Análisis estadístico detallado**: Niveles de confianza basados en múltiples métricas
- **Reportes visuales comprensivos**: Dashboard de 6 paneles con análisis completo
- **Documentación completa**: Manuales en español e interpretación de resultados

<img src="https://raw.githubusercontent.com/alejoduque/ID_indv/refs/heads/main/ocelot_enhanced_comparison.png" /> <br>

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/alejoduque/ID_indv.git
cd ID_indv

# Crear y activar entorno virtual (recomendado)
python3 -m venv ocelot_env
source ocelot_env/bin/activate  # macOS/Linux
# ocelot_env\Scripts\activate   # Windows

# Instalar dependencias
pip install opencv-python matplotlib numpy
```

## 💻 Uso Básico

### 🎥 Comparación de Videos (Recomendado)
**El sistema acepta las rutas de los videos como argumentos de línea de comandos:**

```bash
# Activar el entorno virtual
source ocelot_env/bin/activate

# Sintaxis: python3 ocelot_video_comparison.py <ruta_video1> <ruta_video2>
python3 ocelot_video_comparison.py video1.mov video2.mov

# Ejemplo con videos de prueba incluidos
python3 ocelot_video_comparison.py 1.mov 2.mov

# Ejemplo con rutas completas
python3 ocelot_video_comparison.py "/ruta/completa/ocelote1.mov" "/ruta/completa/ocelote2.mp4"
```

**Formatos de video soportados:** .mov, .mp4, .avi, .mkv y otros formatos compatibles con OpenCV

### 📷 Comparación de Imágenes Estáticas
**Para imágenes estáticas, también se pueden pasar como argumentos:**

```bash
# Sintaxis: python3 ocelot_pattern_comparison.py <imagen1> <imagen2>
python3 ocelot_pattern_comparison.py imagen1.jpg imagen2.jpg

# O usar las imágenes predeterminadas (sin argumentos)
python3 ocelot_pattern_comparison.py
# (Busca automáticamente Ocelote_compare_1.jpg y Ocelote_compare_2.jpg)
```

### 📊 Archivos Generados
Después de ejecutar el análisis, se generan:
- `ocelot_pattern_profile_analysis.png` - **Dashboard de 6 paneles con análisis completo**
- `ocelot_enhanced_pattern_matches.png` - **Mejores coincidencias con detalles de patrones**
- `ocelot_analysis_results.json` - **Métricas detalladas y estadísticas**

## 📊 Resultados Demostrados

### 🧪 Caso de Estudio: Videos de Prueba
- **Videos analizados**: `1.mov` vs `2.mov` (videos cortos de 3+ segundos)
- **Resultado**: **CONFIANZA MODERADA-ALTA - PROBABLEMENTE EL MISMO INDIVIDUO** (75-85%)
- **Análisis de patrones**: 49.7% de similaridad máxima de patrones específicos
- **Evidencia detallada**:
  - **Rosetas detectadas**: Hasta 52.3% de coincidencia con 96 matches fuertes
  - **Manchas sólidas**: 44.0% promedio con 108 detecciones
  - **Puntuación ponderada**: 549.0 (muy fuerte)
  - **Comparaciones totales**: 108 combinaciones de fotogramas (9x12)

### 🎯 Interpretación de Niveles de Confianza
- **90-95%**: ALTA CONFIANZA - Mismo individuo
- **75-85%**: CONFIANZA MODERADA-ALTA - Probablemente el mismo individuo
- **55-70%**: CONFIANZA MODERADA - Posiblemente el mismo individuo
- **35-50%**: CONFIANZA BAJA-MODERADA - Incierto
- **15-35%**: CONFIANZA BAJA - Probablemente individuos diferentes

## 📁 Archivos Incluidos

```
├── ocelot_video_comparison.py                 # 🆕 Script principal mejorado con detección de patrones específicos
├── ocelot_pattern_comparison.py               # Script para imágenes estáticas
├── OCELOT_IDENTIFICATION_GUIDE.md            # 🆕 Guía completa de uso en inglés
├── RESUMEN_ANALISIS_OCELOTE.md               # Resultados detallados en español
├── MANUAL_USO_SISTEMA_OCELOTE.md             # Manual completo de usuario
├── ocelot_analysis_results.json              # 🆕 Resultados con métricas de patrones mejoradas
├── ocelot_pattern_profile_analysis.png       # 🆕 Dashboard de 6 paneles con análisis visual completo
├── ocelot_enhanced_pattern_matches.png       # 🆕 Visualización mejorada con detalles de patrones
├── ocelot_best_matches.png                   # Visualización tradicional de coincidencias
└── README.md                                 # 🆕 Este archivo actualizado
```

## 🔬 Metodología Científica Mejorada

### 🆕 **Pipeline de Análisis de Patrones Específicos**
1. **Extracción de fotogramas**: 12-15 fotogramas distribuidos uniformemente por video
2. **Mejoramiento de imagen**: CLAHE, filtrado bilateral, detección de bordes Canny
3. **Detección de patrones de ocelote**:
   - **Rosetas**: HoughCircles con validación morfológica centro-borde
   - **Manchas sólidas**: Blob detection con filtros de circularidad y convexidad
   - **Validación**: Verificación de contraste y regularidad de patrones
4. **Comparación espacial**: Matching basado en posición, tamaño y confianza
5. **Extracción tradicional**: Algoritmos SIFT (invariante a escala) y ORB (robusto)
6. **Análisis estadístico**: Comparación cruzada de todas las combinaciones de fotogramas
7. **Evaluación ponderada**: Sistema de puntuación que prioriza patrones específicos (2x peso)

### 📊 **Sistema de Visualización Analítica**
- **Matrices de similaridad**: Heatmaps separados para rosetas, manchas y patrones generales
- **Líneas de perfil**: Tendencias de coincidencia a través de todos los fotogramas
- **Análisis de consistencia**: Varianza de patrones para evaluar estabilidad
- **Dashboard resumido**: Métricas consolidadas con indicadores de confianza

## 📈 Aplicaciones en Conservación

- **Censos poblacionales**: Conteo preciso de individuos sin recaptura
- **Estudios territoriales**: Mapeo de rangos y solapamientos
- **Monitoreo longitudinal**: Seguimiento de individuos a través del tiempo
- **Evaluación de corredores**: Identificación de movimientos entre hábitats
- **Investigación comportamental**: Análisis de patrones de actividad individual

## 🎓 Validación Científica

### 🆕 **Fortalezas del Sistema Mejorado**
- ✅ **Análisis específico de ocelote**: Detección especializada de rosetas y patrones únicos de la especie
- ✅ **Visualización comprensiva**: Dashboard de 6 paneles con análisis visual completo
- ✅ **Múltiples métricas**: Combina patrones específicos con características tradicionales
- ✅ **Puntuación ponderada**: Prioriza evidencia de patrones de piel sobre similaridad general
- ✅ **Validación cruzada**: Múltiples algoritmos y comparaciones estadísticas
- ✅ **Interfaz flexible**: Acepta rutas de video por línea de comandos
- ✅ **Reproducibilidad**: Metodología estandarizada y completamente documentada

### ⚠️ **Limitaciones Conocidas**
- Requiere patrones de piel laterales claramente visibles
- Dependiente de calidad de iluminación (infrarroja o natural)
- Optimizado específicamente para patrones de ocelote y felinos similares
- Mejor rendimiento con videos de 3+ segundos de duración

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Hacer fork del repositorio
2. Crear una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo `LICENSE` para detalles.

## 📞 Contacto y Soporte

**Proyecto MANAKAI - Monitoreo de Fauna con Cámaras Trampa**

- **Documentación técnica**: Ver `MANUAL_USO_SISTEMA_OCELOTE.md`
- **Resultados científicos**: Ver `RESUMEN_ANALISIS_OCELOTE.md`
- **Issues**: Usar el sistema de issues de GitHub para reportar problemas

## 🏆 Reconocimientos

Desarrollado inicialmente como parte del Proyecto MANAKAI para conservación de fauna neotropical mediante tecnologías no invasivas de monitoreo.

---

**Última actualización**: Septiembre 2025  
**Versión**: 2.0 - Sistema Mejorado con Análisis de Patrones Específicos  

**Estado**: Producción - Completamente funcional con mejoras significativas 


