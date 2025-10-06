# TMDB Movie Data Collector

## Integrantes

- Wischñevsky David
- Vilamowski Abril
- Liu Jonathan


Un proyecto para recolectar datos de películas de la API de TMDB y almacenarlos en una base de datos para análisis de datos y data science.

## Características

- Recolección masiva de datos de películas desde TMDB API
- Almacenamiento en base de datos PostgreSQL
- Configuración flexible mediante variables de entorno
- Logging detallado de operaciones
- Procesamiento por lotes para optimizar rendimiento
- Soporte para múltiples tipos de datos (películas, géneros, créditos, etc.)

## Instalación

### Opción 1: Instalación Local

1. Clona el repositorio:
```bash
git clone <tu-repositorio>
cd tmdb-movie-collector
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las variables de entorno:
```bash
cp env.example .env
```

4. Edita el archivo `.env` con tus credenciales:
   - Obtén tu API key de TMDB en: https://www.themoviedb.org/settings/api
   - Configura los parámetros de tu base de datos PostgreSQL

### Opción 2: Instalación con Docker (Recomendado)

1. Clona el repositorio:
```bash
git clone <tu-repositorio>
cd tmdb-movie-collector
```

2. Configura las variables de entorno:
```bash
cp env.example .env
# Edita .env con tu API key de TMDB
```

3. Inicia la base de datos con Docker:
```bash
./scripts/docker-setup.sh start-db
```

4. Ejecuta el recolector:
```bash
# Opción A: Solo base de datos (ejecutas Python localmente)
./scripts/docker-setup.sh start-db
python src/main.py --init-db --popular --max-pages 10

# Opción B: Todo en Docker
./scripts/docker-setup.sh run-collector-custom --init-db --popular --max-pages 10
```

## Configuración

### Variables de Entorno

- `TMDB_API_KEY`: Tu API key de TMDB
- `TMDB_BASE_URL`: URL base de la API de TMDB
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: Configuración de la base de datos
- `BATCH_SIZE`: Número de películas a procesar por lote
- `MAX_PAGES`: Número máximo de páginas a recolectar
- `LANGUAGE`, `REGION`: Configuración regional y de idioma

## Uso

### Con Docker (Recomendado)

```bash
# Iniciar base de datos
./scripts/docker-setup.sh start-db

# Ejecutar recolector principal (todo en Docker)
./scripts/docker-setup.sh run-collector-custom --init-db --popular

# Recolectar solo géneros
./scripts/docker-setup.sh run-collector-custom --genres-only

# Recolectar películas populares
./scripts/docker-setup.sh run-collector-custom --popular --max-pages 10

# Ver estado de contenedores
./scripts/docker-setup.sh status
```

### Sin Docker

1. Ejecuta el recolector principal:
```bash
python src/main.py
```

2. Para recolectar solo géneros:
```bash
python src/collectors/genre_collector.py
```

3. Para recolectar películas populares:
```bash
python src/collectors/movie_collector.py
```

## Estructura del Proyecto

```
├── src/
│   ├── main.py                 # Punto de entrada principal
│   ├── config.py              # Configuración de la aplicación
│   ├── database/
│   │   ├── models.py          # Modelos de base de datos
│   │   └── connection.py      # Conexión a la base de datos
│   ├── collectors/
│   │   ├── movie_collector.py # Recolector de películas
│   │   ├── genre_collector.py # Recolector de géneros
│   │   └── credit_collector.py # Recolector de créditos
│   └── utils/
│       ├── api_client.py      # Cliente para TMDB API
│       └── logger.py          # Configuración de logging
├── scripts/
│   ├── docker-setup.sh        # Script de gestión de Docker
│   ├── setup_database.sql     # Script de configuración de BD
│   └── init.sql               # Script de inicialización de BD
├── logs/                      # Archivos de log
├── data/                      # Datos exportados
├── docker-compose.yml         # Configuración Docker
├── Dockerfile                 # Imagen de la aplicación
├── requirements.txt           # Dependencias de Python
├── requirements-analysis.txt  # Dependencias para análisis
├── env.example               # Ejemplo de variables de entorno
├── DOCKER.md                 # Documentación de Docker
└── README.md                 # Este archivo
```

## Base de Datos

El proyecto crea las siguientes tablas:
- `movies`: Información básica de películas
- `genres`: Géneros cinematográficos
- `movie_genres`: Relación entre películas y géneros
- `credits`: Créditos de películas (actores, directores)
- `keywords`: Palabras clave de películas

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
