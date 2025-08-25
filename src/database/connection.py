from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
from src.config import Config
from src.database.models import Base

class DatabaseManager:
    """Gestor de conexiones y sesiones de base de datos."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Configura el motor de base de datos."""
        try:
            database_url = Config.get_database_url()
            self.engine = create_engine(
                database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Motor de base de datos configurado correctamente")
            
        except Exception as e:
            logger.error(f"Error configurando el motor de base de datos: {str(e)}")
            raise
    
    def create_tables(self):
        """Crea todas las tablas definidas en los modelos."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tablas creadas correctamente")
        except SQLAlchemyError as e:
            logger.error(f"Error creando tablas: {str(e)}")
            raise
    
    def drop_tables(self):
        """Elimina todas las tablas (¡CUIDADO!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("Todas las tablas han sido eliminadas")
        except SQLAlchemyError as e:
            logger.error(f"Error eliminando tablas: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Retorna una nueva sesión de base de datos."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call setup_engine() first.")
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Conexión a la base de datos exitosa")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error conectando a la base de datos: {str(e)}")
            return False
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.engine:
            self.engine.dispose()
            logger.info("Conexión a la base de datos cerrada")

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()

def get_db_session() -> Session:
    """Función helper para obtener una sesión de base de datos."""
    return db_manager.get_session()

def init_database():
    """Inicializa la base de datos creando las tablas."""
    if db_manager.test_connection():
        db_manager.create_tables()
        return True
    return False
