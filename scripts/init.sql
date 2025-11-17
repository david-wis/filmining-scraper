-- Script de inicialización para PostgreSQL en Docker
-- Este script se ejecuta automáticamente cuando el contenedor se inicia por primera vez

-- Crear extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Configurar parámetros de rendimiento para desarrollo
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Recargar configuración
SELECT pg_reload_conf();

-- Crear índices adicionales para optimizar consultas de análisis
-- (Estos se crearán automáticamente cuando se ejecute el recolector)

-- Configurar logging para desarrollo
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- Recargar configuración de logging
SELECT pg_reload_conf();

-- Crear un usuario adicional para análisis (opcional)
-- CREATE USER analyst_user WITH PASSWORD 'analyst_password';
-- GRANT CONNECT ON DATABASE movie_database TO analyst_user;
-- GRANT USAGE ON SCHEMA public TO analyst_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analyst_user;
