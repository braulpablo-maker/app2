# 📉 Análisis de Declinación de Pozos

Una aplicación web interactiva para el análisis de declinación de producción de pozos petroleros utilizando el modelo de Arps. Construida con Streamlit, permite cargar datos de producción, ajustar curvas de declinación, generar pronósticos y comparar con datos de control.

## 🚀 Características

- **Carga de Datos**: Soporte para archivos CSV con detección automática de columnas (fecha, pozo, producción)
- **Análisis de Declinación**: Ajuste automático de curvas hiperbólicas de Arps para múltiples pozos
- **Visualización Interactiva**: Gráficos de Plotly para datos históricos, ajustes y pronósticos
- **Pronósticos**: Generación de pronósticos de producción con métodos continuo y ajustado
- **Comparación de Control**: Validación de pronósticos contra datos reales de control
- **Interfaz Moderna**: Diseño inspirado en iPhone con gradientes y elementos visuales atractivos
- **Exportación**: Descarga de parámetros ajustados y pronósticos en formato CSV

## 📋 Requisitos del Sistema

- Python 3.8+
- Dependencias listadas en `requirements.txt`

## 🛠️ Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd <directorio-del-proyecto>
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Uso

### Ejecución de la Aplicación

```bash
streamlit run declinacion.py
```

La aplicación se abrirá en tu navegador web predeterminado.

### Carga de Datos

- **Subir Archivo CSV**: Utiliza el uploader en la barra lateral para cargar tu propio archivo CSV
- **Dataset de Ejemplo**: Selecciona uno de los archivos en la carpeta `Datos/` para probar la aplicación

El formato esperado del CSV es:
- Columna de fecha (ej: `FECHA[DD/MM/YYYY]`)
- Columna de identificación de pozo (ej: `POZO_id`)
- Columna de producción (ej: `qo[m3/DC]`)

### Análisis de Declinación

1. Carga los datos de producción
2. Ve a la pestaña "📉 Declinación"
3. Haz clic en "Ajustar todos los pozos" para ajuste automático, o selecciona un pozo individual
4. Ajusta el periodo de análisis con el selector de rango mensual
5. Usa "Vista previa del ajuste" para revisar antes de guardar
6. Guarda el ajuste para usarlo en pronósticos

### Generación de Pronósticos

1. Ve a la pestaña "🔮 Pronóstico"
2. Selecciona los pozos deseados
3. Configura la fecha de inicio y duración del pronóstico
4. Elige el método de proyección:
   - **Continuo**: Continúa la curva ajustada
   - **Desde último valor**: Ajusta basado en el último dato histórico
5. Visualiza y descarga los pronósticos

### Comparación de Control

1. Ve a la pestaña "📋 Control"
2. Sube un archivo CSV con datos de control (formato similar a producción)
3. Configura el umbral de alerta (%)
4. Revisa las comparaciones y alertas automáticas

## 📁 Estructura del Proyecto

```
.
├── declinacion.py          # Script principal de la aplicación
├── requirements.txt        # Dependencias de Python
├── README.md              # Este archivo
├── Datos/                 # Carpeta con datasets de ejemplo
│   ├── ProduccionB.csv    # Datos históricos de producción
│   └── UltimoControlB.csv # Datos de control para comparación
└── tests/                 # Carpeta para tests (vacía actualmente)
```

## 🔧 Dependencias

- **streamlit**: Framework web para aplicaciones de datos
- **pandas**: Manipulación y análisis de datos
- **numpy**: Computación numérica
- **scipy**: Optimización y ajuste de curvas
- **plotly**: Visualización interactiva de datos

## 📖 Modelo de Arps

La aplicación utiliza el modelo hiperbólico de Arps para el análisis de declinación:

```
q(t) = q₀ / (1 + b * D * t)^(1/b)
```

Donde:
- `q(t)`: Producción en el tiempo t
- `q₀`: Producción inicial
- `D`: Tasa de declinación
- `b`: Exponente hiperbólico (0 ≤ b ≤ 5)

## 🎨 Interfaz de Usuario

- Diseño moderno con gradientes y sombras
- Tipografía San Francisco (inspirada en iOS)
- Elementos interactivos con feedback visual
- Layout responsivo optimizado para desktop

## 📤 Exportación de Resultados

- **Parámetros de Ajuste**: CSV con q₀, D, b, MSE, R² y periodos de ajuste
- **Pronósticos**: CSV detallado con fechas y valores pronosticados
- **Comparaciones de Control**: CSV con deltas, porcentajes y alertas

## 🐛 Solución de Problemas

### Errores Comunes

1. **Columnas no detectadas**: Si el CSV tiene nombres de columnas no estándar, la aplicación pedirá mapeo manual
2. **Ajuste fallido**: Asegúrate de tener al menos 3 puntos de datos válidos (> 0.1 m³/d)
3. **Codificación de archivos**: Soporte para UTF-8, UTF-8-BOM y Latin-1

### Logs y Debugging

La aplicación maneja errores gracefully y muestra mensajes informativos. Para debugging avanzado, revisa la consola del navegador.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

---

**Nota**: Esta aplicación está diseñada para análisis de datos petroleros y no constituye asesoramiento técnico profesional. Siempre valida los resultados con expertos del dominio.