#!/bin/bash

# Script para configurar y gestionar el entorno Docker para TMDB Movie Collector

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con colores
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  TMDB Movie Collector Docker${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Función para verificar si Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado. Por favor instala Docker primero."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose no está instalado. Por favor instala Docker Compose primero."
        exit 1
    fi
    
    print_message "Docker y Docker Compose están instalados correctamente."
}

# Función para verificar si el archivo .env existe
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning "El archivo .env no existe. Creando desde env.example..."
        cp env.example .env
        print_message "Archivo .env creado. Por favor edítalo con tus credenciales antes de continuar."
        print_message "Especialmente necesitas configurar:"
        print_message "  - TMDB_API_KEY: Tu API key de TMDB"
        print_message "  - DB_USER y DB_PASSWORD: Credenciales de la base de datos"
        exit 1
    fi
}

# Función para iniciar solo la base de datos
start_database() {
    print_message "Iniciando base de datos PostgreSQL..."
    docker-compose up -d postgres pgadmin
    
    print_message "Esperando a que la base de datos esté lista..."
    sleep 10
    
    print_message "Base de datos iniciada correctamente!"
    print_message "PostgreSQL disponible en: localhost:5432"
    print_message "pgAdmin disponible en: http://localhost:8080"
    print_message "  - Email: admin@tmdb.com"
    print_message "  - Password: admin123"
}

# Función para detener la base de datos
stop_database() {
    print_message "Deteniendo base de datos..."
    docker-compose down
    print_message "Base de datos detenida."
}

# Función para reiniciar la base de datos
restart_database() {
    print_message "Reiniciando base de datos..."
    docker-compose restart postgres pgadmin
    print_message "Base de datos reiniciada."
}

# Función para ver logs de la base de datos
logs_database() {
    print_message "Mostrando logs de la base de datos..."
    docker-compose logs -f postgres
}

# Función para ejecutar el recolector en modo desarrollo
run_collector_dev() {
    print_message "Ejecutando recolector en modo desarrollo..."
    docker-compose -f docker-compose.dev.yml up --build movie-collector
}

# Función para ejecutar el recolector con argumentos personalizados
run_collector_custom() {
    local args="$@"
    if [ -z "$args" ]; then
        print_error "Debes proporcionar argumentos para el recolector."
        print_message "Ejemplos:"
        print_message "  $0 run-collector --init-db --popular --max-pages 10"
        print_message "  $0 run-collector --genres-only"
        exit 1
    fi
    
    print_message "Ejecutando recolector con argumentos: $args"
    docker-compose -f docker-compose.dev.yml run --rm movie-collector python src/main.py $args
}

# Función para limpiar todo
cleanup() {
    print_warning "Esto eliminará todos los contenedores, volúmenes y datos."
    read -p "¿Estás seguro? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "Limpiando todo..."
        docker-compose down -v --remove-orphans
        docker-compose -f docker-compose.dev.yml down -v --remove-orphans
        docker system prune -f
        print_message "Limpieza completada."
    else
        print_message "Limpieza cancelada."
    fi
}

# Función para mostrar el estado
status() {
    print_message "Estado de los contenedores:"
    docker-compose ps
    
    echo
    print_message "Volúmenes:"
    docker volume ls | grep tmdb
    
    echo
    print_message "Redes:"
    docker network ls | grep movie
}

# Función para mostrar ayuda
show_help() {
    print_header
    echo
    echo "Uso: $0 [COMANDO]"
    echo
    echo "Comandos disponibles:"
    echo "  start-db          - Iniciar solo la base de datos"
    echo "  stop-db           - Detener la base de datos"
    echo "  restart-db        - Reiniciar la base de datos"
    echo "  logs-db           - Mostrar logs de la base de datos"
    echo "  run-dev           - Ejecutar recolector en modo desarrollo"
    echo "  run-collector     - Ejecutar recolector con argumentos personalizados"
    echo "  status            - Mostrar estado de contenedores"
    echo "  cleanup           - Limpiar todo (contenedores, volúmenes, datos)"
    echo "  help              - Mostrar esta ayuda"
    echo
    echo "Ejemplos:"
    echo "  $0 start-db"
    echo "  $0 run-collector --init-db --popular --max-pages 10"
    echo "  $0 run-collector --genres-only"
    echo "  $0 status"
}

# Función principal
main() {
    case "${1:-help}" in
        start-db)
            check_docker
            check_env_file
            start_database
            ;;
        stop-db)
            check_docker
            stop_database
            ;;
        restart-db)
            check_docker
            restart_database
            ;;
        logs-db)
            check_docker
            logs_database
            ;;
        run-dev)
            check_docker
            check_env_file
            run_collector_dev
            ;;
        run-collector)
            check_docker
            check_env_file
            shift
            run_collector_custom "$@"
            ;;
        status)
            check_docker
            status
            ;;
        cleanup)
            check_docker
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Comando desconocido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@"
