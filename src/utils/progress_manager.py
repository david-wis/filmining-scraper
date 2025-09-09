import json
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_db_session
from src.database.models import ImportProgress
from src.config import Config

class ProgressManager:
    """Gestor de progreso para importaciones con capacidad de reanudación."""
    
    def __init__(self):
        self.current_progress: Optional[ImportProgress] = None
        self.session: Optional[Session] = None
        self.logger = logger.bind(name="ProgressManager")
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Configura manejadores de señales para guardar progreso al interrumpir."""
        def signal_handler(signum, frame):
            self.logger.info(f"Señal {signum} recibida. Guardando progreso...")
            self.pause_current_import()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_import(self, import_type: str, endpoint: str, max_pages: int = None) -> ImportProgress:
        """Inicia una nueva importación o reanuda una existente."""
        self.session = get_db_session()
        
        try:
            # Buscar importación existente del mismo tipo
            existing_progress = self.session.query(ImportProgress).filter_by(
                import_type=import_type,
                endpoint=endpoint,
                status='running'
            ).first()
            
            if existing_progress:
                self.logger.info(f"Reanudando importación existente: {existing_progress}")
                self.current_progress = existing_progress
                return existing_progress
            
            # Crear nueva importación
            config_snapshot = {
                'batch_size': Config.BATCH_SIZE,
                'max_pages': max_pages or Config.MAX_PAGES,
                'language': Config.LANGUAGE,
                'region': Config.REGION,
                'collect_genres': Config.COLLECT_GENRES,
                'collect_credits': Config.COLLECT_CREDITS,
                'collect_keywords': Config.COLLECT_KEYWORDS
            }
            
            new_progress = ImportProgress(
                import_type=import_type,
                endpoint=endpoint,
                current_page=1,
                total_pages=max_pages or Config.MAX_PAGES,
                status='running',
                config_snapshot=json.dumps(config_snapshot)
            )
            
            self.session.add(new_progress)
            self.session.commit()
            
            self.current_progress = new_progress
            self.logger.info(f"Nueva importación iniciada: {new_progress}")
            return new_progress
            
        except Exception as e:
            self.logger.error(f"Error iniciando importación: {str(e)}")
            if self.session:
                self.session.rollback()
            raise
    
    def update_progress(self, page: int, movies_processed: int = 0, 
                       movies_new: int = 0, movies_updated: int = 0, 
                       errors_count: int = 0):
        """Actualiza el progreso de la importación actual."""
        if not self.current_progress:
            return
        
        try:
            self.current_progress.current_page = page
            self.current_progress.movies_processed += movies_processed
            self.current_progress.movies_new += movies_new
            self.current_progress.movies_updated += movies_updated
            self.current_progress.errors_count += errors_count
            self.current_progress.last_updated = datetime.now()
            
            # Calcular estimación de finalización
            if page > 1 and movies_processed > 0:
                self._calculate_estimated_completion()
            
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error actualizando progreso: {str(e)}")
            self.session.rollback()
    
    def _calculate_estimated_completion(self):
        """Calcula la estimación de tiempo de finalización."""
        if not self.current_progress or self.current_progress.current_page <= 1:
            return
        
        try:
            # Calcular velocidad promedio (páginas por minuto)
            elapsed_time = datetime.now() - self.current_progress.started_at
            elapsed_minutes = elapsed_time.total_seconds() / 60
            
            if elapsed_minutes > 0:
                pages_per_minute = (self.current_progress.current_page - 1) / elapsed_minutes
                remaining_pages = self.current_progress.total_pages - self.current_progress.current_page
                
                if pages_per_minute > 0:
                    remaining_minutes = remaining_pages / pages_per_minute
                    estimated_completion = datetime.now() + timedelta(minutes=remaining_minutes)
                    self.current_progress.estimated_completion = estimated_completion
                    
        except Exception as e:
            self.logger.debug(f"Error calculando estimación: {str(e)}")
    
    def complete_import(self, success: bool = True, error_message: str = None):
        """Marca la importación como completada o fallida."""
        if not self.current_progress:
            return
        
        try:
            if success:
                self.current_progress.status = 'completed'
                self.current_progress.completed_at = datetime.now()
                self.logger.info(f"Importación completada exitosamente: {self.current_progress}")
            else:
                self.current_progress.status = 'failed'
                self.current_progress.error_message = error_message
                self.logger.error(f"Importación fallida: {error_message}")
            
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error completando importación: {str(e)}")
            self.session.rollback()
        finally:
            self.current_progress = None
    
    def pause_current_import(self):
        """Pausa la importación actual."""
        if not self.current_progress:
            return
        
        try:
            self.current_progress.status = 'paused'
            self.current_progress.last_updated = datetime.now()
            self.session.commit()
            self.logger.info(f"Importación pausada: {self.current_progress}")
            
        except Exception as e:
            self.logger.error(f"Error pausando importación: {str(e)}")
            self.session.rollback()
    
    def get_import_status(self, import_id: int = None) -> Optional[ImportProgress]:
        """Obtiene el estado de una importación específica o la actual."""
        session = get_db_session()
        
        try:
            if import_id:
                return session.query(ImportProgress).filter_by(id=import_id).first()
            else:
                return self.current_progress
                
        except Exception as e:
            self.logger.error(f"Error obteniendo estado de importación: {str(e)}")
            return None
        finally:
            session.close()
    
    def list_imports(self, status: str = None) -> List[ImportProgress]:
        """Lista todas las importaciones, opcionalmente filtradas por estado."""
        session = get_db_session()
        
        try:
            query = session.query(ImportProgress)
            if status:
                query = query.filter_by(status=status)
            
            return query.order_by(ImportProgress.started_at.desc()).all()
            
        except Exception as e:
            self.logger.error(f"Error listando importaciones: {str(e)}")
            return []
        finally:
            session.close()
    
    def cleanup_failed_imports(self) -> int:
        """Limpia importaciones fallidas o muy antiguas."""
        session = get_db_session()
        
        try:
            # Eliminar importaciones fallidas de hace más de 7 días
            cutoff_date = datetime.now() - timedelta(days=7)
            
            failed_imports = session.query(ImportProgress).filter(
                ImportProgress.status == 'failed',
                ImportProgress.started_at < cutoff_date
            ).all()
            
            count = len(failed_imports)
            for import_progress in failed_imports:
                session.delete(import_progress)
            
            session.commit()
            self.logger.info(f"Limpiadas {count} importaciones fallidas antiguas")
            return count
            
        except Exception as e:
            self.logger.error(f"Error limpiando importaciones: {str(e)}")
            session.rollback()
            return 0
        finally:
            session.close()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del progreso actual."""
        if not self.current_progress:
            return {}
        
        progress = self.current_progress
        total_pages = progress.total_pages or 1
        current_page = progress.current_page
        
        return {
            'import_id': progress.id,
            'import_type': progress.import_type,
            'endpoint': progress.endpoint,
            'current_page': current_page,
            'total_pages': total_pages,
            'progress_percentage': (current_page / total_pages) * 100,
            'movies_processed': progress.movies_processed,
            'movies_new': progress.movies_new,
            'movies_updated': progress.movies_updated,
            'errors_count': progress.errors_count,
            'status': progress.status,
            'started_at': progress.started_at,
            'last_updated': progress.last_updated,
            'estimated_completion': progress.estimated_completion,
            'elapsed_time': datetime.now() - progress.started_at if progress.started_at else None
        }
    
    def close(self):
        """Cierra la sesión de base de datos."""
        if self.session:
            self.session.close()
            self.session = None
