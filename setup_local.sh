#!/bin/bash

# Script para configurar el entorno local de TMDB Movie Collector
# Usa Docker para la base de datos y Python local para la aplicación

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
    echo -e "${BLUE}  TMDB Movie Collector Setup${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Función para verificar si Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado. Por favor instala Docker primero."
        print_message "Puedes instalarlo desde: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose plugin no está disponible. Por favor instala Docker Compose plugin primero."
        exit 1
    fi
    
    print_message "Docker y Docker Compose están instalados correctamente."
}

# Función para verificar si Python está instalado
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 no está instalado. Por favor instala Python 3.8+ primero."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Se requiere Python 3.8 o superior. Versión actual: $python_version"
        exit 1
    fi
    
    print_message "Python $python_version detectado correctamente."
}

# Función para crear entorno virtual
create_venv() {
    if [ -d "venv" ]; then
        print_warning "El entorno virtual 'venv' ya existe."
        read -p "¿Quieres recrearlo? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message "Eliminando entorno virtual existente..."
            rm -rf venv
        else
            print_message "Usando entorno virtual existente."
            return 0
        fi
    fi
    
    print_message "Creando entorno virtual..."
    python3 -m venv venv
    print_message "Entorno virtual creado exitosamente."
}

# Función para instalar dependencias sin psycopg2
install_dependencies() {
    print_message "Instalando dependencias (sin psycopg2)..."
    source venv/bin/activate
    pip install --upgrade pip
    
    # Instalar dependencias principales sin psycopg2
    pip install requests python-dotenv sqlalchemy pandas numpy tqdm python-dateutil pydantic loguru
    
    print_message "Dependencias instaladas correctamente."
}

# Función para configurar variables de entorno
setup_env() {
    if [ ! -f ".env" ]; then
        print_message "Creando archivo .env desde env.example..."
        cp env.example .env
        print_warning "Archivo .env creado. Por favor edítalo con tu API key de TMDB."
        print_message "Especialmente necesitas configurar:"
        print_message "  - TMDB_API_KEY: Tu API key de TMDB"
    else
        print_message "Archivo .env ya existe."
    fi
}

# Función para iniciar base de datos con Docker
start_database() {
    print_message "Iniciando base de datos PostgreSQL con Docker..."
    
    # Verificar si Docker está corriendo
    if ! docker info &> /dev/null; then
        print_error "Docker no está corriendo. Por favor inicia Docker primero."
        exit 1
    fi
    
    # Iniciar solo la base de datos
    docker compose up -d postgres pgadmin
    
    print_message "Esperando a que la base de datos esté lista..."
    sleep 10
    
    print_message "Base de datos iniciada correctamente!"
    print_message "PostgreSQL disponible en: localhost:5432"
    print_message "pgAdmin disponible en: http://localhost:8080"
    print_message "  - Email: admin@tmdb.com"
    print_message "  - Password: admin123"
}

# Función para crear scripts útiles
create_scripts() {
    # Script para activar el entorno virtual
    cat > activate.sh << 'EOF'
#!/bin/bash
# Script para activar el entorno virtual

if [ ! -d "venv" ]; then
    echo "Error: El entorno virtual 'venv' no existe."
    echo "Ejecuta ./setup_local.sh primero."
    exit 1
fi

source venv/bin/activate
echo "Entorno virtual activado."
echo "Para desactivar, ejecuta: deactivate"
EOF
    
    chmod +x activate.sh
    
    # Script para ejecutar la aplicación
    cat > run.sh << 'EOF'
#!/bin/bash
# Script para ejecutar la aplicación

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "Error: El entorno virtual 'venv' no existe."
    echo "Ejecuta ./setup_local.sh primero."
    exit 1
fi

# Activar entorno virtual
source venv/bin/activate

# Ejecutar la aplicación con los argumentos proporcionados
python src/main.py "$@"
EOF
    
    chmod +x run.sh
    
    # Script para iniciar/detener la base de datos
    cat > db.sh << 'EOF'
#!/bin/bash
# Script para gestionar la base de datos

case "${1:-help}" in
    start)
        echo "Iniciando base de datos..."
        docker compose up -d postgres pgadmin
        echo "Base de datos iniciada."
        echo "PostgreSQL: localhost:5432"
        echo "pgAdmin: http://localhost:8080"
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
    help|--help|-h)
        echo "Uso: $0 [COMANDO]"
        echo ""
        echo "Comandos:"
        echo "  start   - Iniciar base de datos"
        echo "  stop    - Detener base de datos"
        echo "  restart - Reiniciar base de datos"
        echo "  logs    - Ver logs"
        echo "  status  - Ver estado"
        echo "  help    - Mostrar esta ayuda"
        ;;
    *)
        echo "Comando desconocido: $1"
        echo "Usa '$0 help' para ver los comandos disponibles."
        exit 1
        ;;
esac
EOF
    
    chmod +x db.sh
    
    print_message "Scripts creados:"
    print_message "  - activate.sh: Activar entorno virtual"
    print_message "  - run.sh: Ejecutar aplicación"
    print_message "  - db.sh: Gestionar base de datos"
}

# Función para mostrar información final
show_final_info() {
    print_header
    echo
    print_message "¡Configuración completada exitosamente!"
    echo
    echo "📁 Entorno virtual: ./venv"
    echo "🔧 Scripts creados:"
    echo "  - ./activate.sh: Activar entorno virtual"
    echo "  - ./run.sh: Ejecutar aplicación"
    echo "  - ./db.sh: Gestionar base de datos"
    echo
    echo "📋 Próximos pasos:"
    echo "1. Configura las variables de entorno:"
    echo "   # Edita .env con tu API key de TMDB"
    echo
    echo "2. Activa el entorno virtual:"
    echo "   source ./activate.sh"
    echo
    echo "3. Inicia la base de datos:"
    echo "   ./db.sh start"
    echo
    echo "4. Ejecuta la aplicación:"
    echo "   ./run.sh --init-db --popular --max-pages 10"
    echo
    echo "🔗 Para más información, consulta README.md"
    echo
    print_warning "Nota: Esta configuración usa Docker para la base de datos y Python local para la aplicación."
    print_warning "Si prefieres usar PostgreSQL local, necesitarás instalar psycopg2 manualmente."
}

# Función para mostrar ayuda
show_help() {
    print_header
    echo
    echo "Uso: $0 [OPCIONES]"
    echo
    echo "Opciones:"
    echo "  --help              - Mostrar esta ayuda"
    echo
    echo "Este script configura:"
    echo "  - Entorno virtual de Python"
    echo "  - Dependencias (sin psycopg2)"
    echo "  - Base de datos PostgreSQL con Docker"
    echo "  - Scripts útiles para gestión"
}

# Función principal
main() {
    case "${1:-setup}" in
        --help|-h)
            show_help
            exit 0
            ;;
        setup)
            print_header
            
            # Verificaciones previas
            check_docker
            check_python
            
            # Configuración
            setup_env
            create_venv
            install_dependencies
            start_database
            create_scripts
            
            # Información final
            show_final_info
            ;;
        *)
            print_error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@"
