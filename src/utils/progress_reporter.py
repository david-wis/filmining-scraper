import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger
from tabulate import tabulate

from src.utils.progress_manager import ProgressManager
from src.database.connection import get_db_session
from src.database.models import ImportProgress, Movie, Genre, Credit, Keyword

class ProgressReporter:
    """Generador de reportes detallados de progreso de importaciones."""
    
    def __init__(self):
        self.progress_manager = ProgressManager()
        self.logger = logger.bind(name="ProgressReporter")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Genera un reporte resumen de todas las importaciones."""
        session = get_db_session()
        
        try:
            # Estadísticas generales
            total_imports = session.query(ImportProgress).count()
            completed_imports = session.query(ImportProgress).filter_by(status='completed').count()
            failed_imports = session.query(ImportProgress).filter_by(status='failed').count()
            running_imports = session.query(ImportProgress).filter_by(status='running').count()
            paused_imports = session.query(ImportProgress).filter_by(status='paused').count()
            
            # Estadísticas de películas
            total_movies = session.query(Movie).count()
            total_genres = session.query(Genre).count()
            total_credits = session.query(Credit).count()
            total_keywords = session.query(Keyword).count()
            
            # Importaciones recientes
            recent_imports = session.query(ImportProgress).order_by(
                ImportProgress.started_at.desc()
            ).limit(10).all()
            
            return {
                'summary': {
                    'total_imports': total_imports,
                    'completed_imports': completed_imports,
                    'failed_imports': failed_imports,
                    'running_imports': running_imports,
                    'paused_imports': paused_imports,
                    'success_rate': (completed_imports / total_imports * 100) if total_imports > 0 else 0
                },
                'data_stats': {
                    'total_movies': total_movies,
                    'total_genres': total_genres,
                    'total_credits': total_credits,
                    'total_keywords': total_keywords
                },
                'recent_imports': [
                    {
                        'id': imp.id,
                        'type': imp.import_type,
                        'status': imp.status,
                        'progress': f"{imp.current_page}/{imp.total_pages}",
                        'movies_processed': imp.movies_processed,
                        'started_at': imp.started_at,
                        'completed_at': imp.completed_at
                    }
                    for imp in recent_imports
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generando reporte resumen: {str(e)}")
            return {}
        finally:
            session.close()
    
    def generate_detailed_report(self, import_id: int) -> Optional[Dict[str, Any]]:
        """Genera un reporte detallado de una importación específica."""
        session = get_db_session()
        
        try:
            import_progress = session.query(ImportProgress).filter_by(id=import_id).first()
            if not import_progress:
                return None
            
            # Calcular métricas de rendimiento
            elapsed_time = None
            estimated_completion = None
            pages_per_hour = 0
            
            if import_progress.started_at:
                if import_progress.completed_at:
                    elapsed_time = import_progress.completed_at - import_progress.started_at
                else:
                    elapsed_time = datetime.now() - import_progress.started_at
                
                if elapsed_time.total_seconds() > 0:
                    pages_per_hour = (import_progress.current_page - 1) / (elapsed_time.total_seconds() / 3600)
                    
                    if import_progress.status == 'running' and pages_per_hour > 0:
                        remaining_pages = import_progress.total_pages - import_progress.current_page
                        estimated_completion = datetime.now() + timedelta(
                            hours=remaining_pages / pages_per_hour
                        )
            
            # Configuración usada
            config = {}
            if import_progress.config_snapshot:
                try:
                    config = json.loads(import_progress.config_snapshot)
                except json.JSONDecodeError:
                    pass
            
            return {
                'import_info': {
                    'id': import_progress.id,
                    'type': import_progress.import_type,
                    'endpoint': import_progress.endpoint,
                    'status': import_progress.status,
                    'started_at': import_progress.started_at,
                    'last_updated': import_progress.last_updated,
                    'completed_at': import_progress.completed_at,
                    'error_message': import_progress.error_message
                },
                'progress': {
                    'current_page': import_progress.current_page,
                    'total_pages': import_progress.total_pages,
                    'progress_percentage': (import_progress.current_page / import_progress.total_pages * 100) if import_progress.total_pages else 0,
                    'movies_processed': import_progress.movies_processed,
                    'movies_new': import_progress.movies_new,
                    'movies_updated': import_progress.movies_updated,
                    'errors_count': import_progress.errors_count
                },
                'performance': {
                    'elapsed_time': elapsed_time,
                    'pages_per_hour': pages_per_hour,
                    'estimated_completion': estimated_completion
                },
                'config': config
            }
            
        except Exception as e:
            self.logger.error(f"Error generando reporte detallado: {str(e)}")
            return None
        finally:
            session.close()
    
    def print_summary_report(self):
        """Imprime un reporte resumen en consola."""
        report = self.generate_summary_report()
        if not report:
            self.logger.error("No se pudo generar el reporte")
            return
        
        print("\n" + "="*60)
        print("📊 REPORTE RESUMEN DE IMPORTACIONES")
        print("="*60)
        
        # Resumen de importaciones
        summary = report['summary']
        print(f"\n📈 ESTADÍSTICAS DE IMPORTACIONES:")
        print(f"   Total de importaciones: {summary['total_imports']}")
        print(f"   ✅ Completadas: {summary['completed_imports']}")
        print(f"   ❌ Fallidas: {summary['failed_imports']}")
        print(f"   🟡 En curso: {summary['running_imports']}")
        print(f"   ⏸️  Pausadas: {summary['paused_imports']}")
        print(f"   📊 Tasa de éxito: {summary['success_rate']:.1f}%")
        
        # Estadísticas de datos
        data_stats = report['data_stats']
        print(f"\n🗄️  DATOS RECOLECTADOS:")
        print(f"   Películas: {data_stats['total_movies']:,}")
        print(f"   Géneros: {data_stats['total_genres']}")
        print(f"   Créditos: {data_stats['total_credits']:,}")
        print(f"   Palabras clave: {data_stats['total_keywords']:,}")
        
        # Importaciones recientes
        recent = report['recent_imports']
        if recent:
            print(f"\n📋 IMPORTACIONES RECIENTES:")
            table_data = []
            for imp in recent:
                status_emoji = "🟢" if imp['status'] == "completed" else "🔴" if imp['status'] == "failed" else "🟡" if imp['status'] == "running" else "⏸️"
                table_data.append([
                    f"{status_emoji} {imp['id']}",
                    imp['type'],
                    imp['status'],
                    imp['progress'],
                    f"{imp['movies_processed']:,}",
                    imp['started_at'].strftime('%Y-%m-%d %H:%M')
                ])
            
            headers = ["ID", "Tipo", "Estado", "Progreso", "Películas", "Iniciado"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\n" + "="*60)
    
    def print_detailed_report(self, import_id: int):
        """Imprime un reporte detallado en consola."""
        report = self.generate_detailed_report(import_id)
        if not report:
            self.logger.error(f"No se encontró importación con ID {import_id}")
            return
        
        print(f"\n" + "="*60)
        print(f"📋 REPORTE DETALLADO - IMPORTACIÓN {import_id}")
        print("="*60)
        
        # Información básica
        info = report['import_info']
        print(f"\n📝 INFORMACIÓN BÁSICA:")
        print(f"   ID: {info['id']}")
        print(f"   Tipo: {info['type']}")
        print(f"   Endpoint: {info['endpoint']}")
        print(f"   Estado: {info['status']}")
        print(f"   Iniciado: {info['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Última actualización: {info['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
        if info['completed_at']:
            print(f"   Completado: {info['completed_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        if info['error_message']:
            print(f"   Error: {info['error_message']}")
        
        # Progreso
        progress = report['progress']
        print(f"\n📊 PROGRESO:")
        print(f"   Páginas: {progress['current_page']}/{progress['total_pages']} ({progress['progress_percentage']:.1f}%)")
        print(f"   Películas procesadas: {progress['movies_processed']:,}")
        print(f"   Películas nuevas: {progress['movies_new']:,}")
        print(f"   Películas actualizadas: {progress['movies_updated']:,}")
        print(f"   Errores: {progress['errors_count']}")
        
        # Rendimiento
        perf = report['performance']
        if perf['elapsed_time']:
            hours = perf['elapsed_time'].total_seconds() / 3600
            print(f"\n⚡ RENDIMIENTO:")
            print(f"   Tiempo transcurrido: {hours:.1f} horas")
            print(f"   Velocidad: {perf['pages_per_hour']:.1f} páginas/hora")
            if perf['estimated_completion']:
                print(f"   Estimación de finalización: {perf['estimated_completion'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Configuración
        config = report['config']
        if config:
            print(f"\n⚙️  CONFIGURACIÓN:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        
        print("\n" + "="*60)
    
    def export_report_to_json(self, filename: str = None) -> str:
        """Exporta el reporte resumen a un archivo JSON."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"import_report_{timestamp}.json"
        
        report = self.generate_summary_report()
        
        # Convertir datetime objects a strings para JSON
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=datetime_converter, ensure_ascii=False)
            
            self.logger.info(f"Reporte exportado a: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exportando reporte: {str(e)}")
            return ""
    
    def close(self):
        """Cierra el gestor de progreso."""
        self.progress_manager.close()

def main():
    """Función principal para ejecutar reportes desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generador de reportes de progreso')
    parser.add_argument('--summary', action='store_true', help='Mostrar reporte resumen')
    parser.add_argument('--detailed', type=int, help='Mostrar reporte detallado de importación específica')
    parser.add_argument('--export', action='store_true', help='Exportar reporte a JSON')
    parser.add_argument('--output', type=str, help='Archivo de salida para exportación')
    
    args = parser.parse_args()
    
    reporter = ProgressReporter()
    
    try:
        if args.summary:
            reporter.print_summary_report()
        
        if args.detailed:
            reporter.print_detailed_report(args.detailed)
        
        if args.export:
            filename = reporter.export_report_to_json(args.output)
            if filename:
                print(f"Reporte exportado a: {filename}")
        
        if not any([args.summary, args.detailed, args.export]):
            reporter.print_summary_report()
            
    finally:
        reporter.close()

if __name__ == "__main__":
    main()
