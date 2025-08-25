#!/bin/bash
# Script para gestionar la base de datos PostgreSQL con Docker

case "${1:-help}" in
    start)
        echo "Iniciando base de datos PostgreSQL..."
        docker compose up -d postgres pgadmin
        echo "Base de datos iniciada."
        echo "PostgreSQL: localhost:5432"
        echo "pgAdmin: http://localhost:8080 (sin credenciales)"
        ;;
    stop)
        echo "Deteniendo base de datos..."
        docker compose down
        echo "Base de datos detenida."
        ;;
    restart)
        echo "Reiniciando base de datos..."
        docker compose restart postgres pgadmin
        echo "Base de datos reiniciada."
        ;;
    logs)
        echo "Mostrando logs de la base de datos..."
        docker compose logs -f postgres
        ;;
    status)
        echo "Estado de la base de datos:"
        docker compose ps
        ;;
    connect)
        echo "Conectando a PostgreSQL..."
        docker exec -it tmdb_movie_db psql -U postgres -d movie_database
        ;;
    help|--help|-h)
        echo "Uso: $0 [COMANDO]"
        echo ""
        echo "Comandos:"
        echo "  start   - Iniciar base de datos"
        echo "  stop    - Detener base de datos"
        echo "  restart - Reiniciar base de datos"
        echo "  logs    - Ver logs"
        echo "  status  - Ver estado"
        echo "  connect - Conectar a PostgreSQL"
        echo "  help    - Mostrar esta ayuda"
        ;;
    *)
        echo "Comando desconocido: $1"
        echo "Usa '$0 help' para ver los comandos disponibles."
        exit 1
        ;;
esac
