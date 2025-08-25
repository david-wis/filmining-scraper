import os
from loguru import logger
from src.config import Config

def setup_logger():
    """Configura el sistema de logging de la aplicación."""
    
    # Crear directorio de logs si no existe
    log_dir = os.path.dirname(Config.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Remover logger por defecto
    logger.remove()
    
    # Configurar logger para consola
    logger.add(
        lambda msg: print(msg, end=""),
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Configurar logger para archivo
    logger.add(
        Config.LOG_FILE,
        level=Config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    return logger

def get_logger(name: str = None):
    """Retorna un logger configurado."""
    if name:
        return logger.bind(name=name)
    return logger
