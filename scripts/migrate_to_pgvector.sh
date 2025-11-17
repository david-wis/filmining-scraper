#!/bin/bash
# Script de migración a pgvector
# Este script te guía paso a paso para migrar tu base de datos a pgvector

set -e  # Salir si hay algún error

echo "=========================================="
echo "Migración a PostgreSQL con pgvector"
echo "=========================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir pasos
print_step() {
    echo -e "${GREEN}[PASO $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[ADVERTENCIA]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    print_error "Docker no está corriendo. Por favor inicia Docker primero."
    exit 1
fi

print_step "1" "Verificando estado actual de la base de datos..."
if docker ps | grep -q tmdb_movie_db; then
    print_warning "La base de datos está corriendo. Necesitamos detenerla para hacer el backup."
    read -p "¿Deseas continuar? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Migración cancelada."
        exit 0
    fi
fi

print_step "2" "Deteniendo contenedores actuales..."
cd "$(dirname "$0")/.."
docker compose down postgres pgadmin 2>/dev/null || true
echo "✓ Contenedores detenidos"

print_step "3" "Haciendo backup de los datos actuales (si existen)..."
if docker volume ls | grep -q "filmining-scraper_postgres_data"; then
    BACKUP_FILE="data/backup_before_migration_$(date +%Y%m%d_%H%M%S).sql"
    echo "Creando backup en: $BACKUP_FILE"
    
    # Iniciar temporalmente el contenedor para hacer backup
    docker compose up -d postgres
    sleep 5
    
    # Esperar a que PostgreSQL esté listo
    echo "Esperando a que PostgreSQL esté listo..."
    for i in {1..30}; do
        if docker exec tmdb_movie_db pg_isready -U postgres > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # Hacer el backup
    docker exec tmdb_movie_db pg_dump -U postgres -d movie_database -F c -f /tmp/backup.dump
    docker cp tmdb_movie_db:/tmp/backup.dump "$BACKUP_FILE"
    docker compose down postgres
    
    echo "✓ Backup creado: $BACKUP_FILE"
else
    print_warning "No se encontró volumen de datos existente. Continuando con migración limpia."
fi

print_step "4" "Eliminando volumen de datos antiguo (para usar nueva imagen)..."
read -p "¿Estás seguro de eliminar el volumen actual? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    docker volume rm filmining-scraper_postgres_data 2>/dev/null || true
    echo "✓ Volumen eliminado"
else
    print_warning "No se eliminó el volumen. La nueva imagen puede no funcionar correctamente."
    print_warning "Si tienes problemas, ejecuta manualmente: docker volume rm filmining-scraper_postgres_data"
fi

print_step "5" "Iniciando PostgreSQL con pgvector..."
docker compose up -d postgres
echo "Esperando a que PostgreSQL esté listo..."
sleep 5

# Esperar a que PostgreSQL esté listo
for i in {1..30}; do
    if docker exec tmdb_movie_db pg_isready -U postgres > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Verificar que pgvector está instalado
if docker exec tmdb_movie_db psql -U postgres -d movie_database -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null 2>&1; then
    echo "✓ PostgreSQL con pgvector iniciado correctamente"
else
    print_error "Error al crear la extensión pgvector"
    exit 1
fi

print_step "6" "Verificando extensión pgvector..."
VECTOR_VERSION=$(docker exec tmdb_movie_db psql -U postgres -d movie_database -t -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" | xargs)
if [ -n "$VECTOR_VERSION" ]; then
    echo "✓ pgvector instalado (versión: $VECTOR_VERSION)"
else
    print_error "pgvector no se instaló correctamente"
    exit 1
fi

print_step "7" "Restaurando datos desde backup.sql..."
if [ -f "data/backup.sql" ]; then
    echo "Restaurando desde: data/backup.sql"
    docker exec -i tmdb_movie_db pg_restore -U postgres -d movie_database --clean --if-exists < data/backup.sql 2>/dev/null || {
        print_warning "pg_restore falló, intentando con psql (puede ser formato SQL plano)..."
        # Si es un dump SQL plano, usar psql
        docker exec -i tmdb_movie_db psql -U postgres -d movie_database < data/backup.sql || {
            print_error "Error al restaurar backup.sql"
            print_warning "Puedes restaurar manualmente con:"
            echo "  docker exec -i tmdb_movie_db pg_restore -U postgres -d movie_database < data/backup.sql"
            exit 1
        }
    }
    echo "✓ Datos restaurados"
else
    print_warning "No se encontró data/backup.sql. Los datos no se restauraron automáticamente."
    echo "Para restaurar manualmente, ejecuta:"
    echo "  docker exec -i tmdb_movie_db pg_restore -U postgres -d movie_database < data/backup.sql"
fi

print_step "8" "Iniciando pgAdmin..."
docker compose up -d pgadmin
echo "✓ pgAdmin iniciado"

echo ""
echo "=========================================="
echo -e "${GREEN}¡Migración completada!${NC}"
echo "=========================================="
echo ""
echo "PostgreSQL con pgvector está corriendo en:"
echo "  - Host: localhost"
echo "  - Puerto: 25432"
echo "  - Base de datos: movie_database"
echo "  - Usuario: postgres"
echo "  - Contraseña: postgres"
echo ""
echo "pgAdmin está disponible en:"
echo "  - http://localhost:8080"
echo "  - Email: admin@tmdb.com"
echo "  - Contraseña: admin123"
echo ""
echo "Para verificar que pgvector funciona, ejecuta:"
echo "  docker exec -it tmdb_movie_db psql -U postgres -d movie_database -c \"SELECT extversion FROM pg_extension WHERE extname = 'vector';\""
echo ""
echo "Para crear una tabla con vectores, puedes usar:"
echo "  CREATE TABLE example (id SERIAL PRIMARY KEY, embedding vector(1536));"
echo ""

