#!/usr/bin/env python3
"""
Ejemplo de análisis de datos con los datos recolectados de TMDB

Este script demuestra cómo usar los datos recolectados para análisis de data science.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
from datetime import datetime, timedelta

from src.database.connection import get_db_session
from src.config import Config

class MovieDataAnalyzer:
    """Analizador de datos de películas recolectadas."""
    
    def __init__(self):
        self.session = get_db_session()
        
    def get_movies_dataframe(self):
        """Obtiene los datos de películas como DataFrame."""
        query = text("""
            SELECT 
                m.id,
                m.tmdb_id,
                m.title,
                m.overview,
                m.release_date,
                m.runtime,
                m.budget,
                m.revenue,
                m.popularity,
                m.vote_average,
                m.vote_count,
                m.adult,
                m.status,
                m.original_language,
                STRING_AGG(g.name, ', ') as genres
            FROM movies m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.id, m.tmdb_id, m.title, m.overview, m.release_date, 
                     m.runtime, m.budget, m.revenue, m.popularity, m.vote_average, 
                     m.vote_count, m.adult, m.status, m.original_language
            ORDER BY m.popularity DESC
        """)
        
        df = pd.read_sql(query, self.session.bind)
        return df
    
    def get_credits_dataframe(self):
        """Obtiene los datos de créditos como DataFrame."""
        query = text("""
            SELECT 
                c.id,
                c.movie_id,
                c.tmdb_person_id,
                c.name,
                c.character,
                c.job,
                c.department,
                c.credit_type,
                c.order,
                m.title as movie_title
            FROM credits c
            JOIN movies m ON c.movie_id = m.id
            ORDER BY c.credit_type, c.order
        """)
        
        df = pd.read_sql(query, self.session.bind)
        return df
    
    def analyze_movie_trends(self):
        """Analiza tendencias en las películas."""
        df = self.get_movies_dataframe()
        
        print("=== ANÁLISIS DE TENDENCIAS DE PELÍCULAS ===")
        print(f"Total de películas: {len(df)}")
        print(f"Rango de fechas: {df['release_date'].min()} a {df['release_date'].max()}")
        print(f"Promedio de calificación: {df['vote_average'].mean():.2f}")
        print(f"Promedio de popularidad: {df['popularity'].mean():.2f}")
        
        # Análisis por década
        df['decade'] = (df['release_date'].dt.year // 10) * 10
        decade_stats = df.groupby('decade').agg({
            'vote_average': 'mean',
            'popularity': 'mean',
            'budget': 'mean',
            'revenue': 'mean'
        }).round(2)
        
        print("\nEstadísticas por década:")
        print(decade_stats)
        
        return df
    
    def analyze_genres(self):
        """Analiza los géneros más populares."""
        df = self.get_movies_dataframe()
        
        # Separar géneros múltiples
        genre_df = df[df['genres'].notna()].copy()
        genre_df['genre_list'] = genre_df['genres'].str.split(', ')
        
        # Expandir géneros
        genre_expanded = []
        for _, row in genre_df.iterrows():
            for genre in row['genre_list']:
                genre_expanded.append({
                    'movie_id': row['id'],
                    'title': row['title'],
                    'genre': genre.strip(),
                    'vote_average': row['vote_average'],
                    'popularity': row['popularity'],
                    'budget': row['budget'],
                    'revenue': row['revenue']
                })
        
        genre_df_expanded = pd.DataFrame(genre_expanded)
        
        # Estadísticas por género
        genre_stats = genre_df_expanded.groupby('genre').agg({
            'vote_average': ['count', 'mean', 'std'],
            'popularity': 'mean',
            'budget': 'mean',
            'revenue': 'mean'
        }).round(2)
        
        print("\n=== ANÁLISIS DE GÉNEROS ===")
        print("Estadísticas por género:")
        print(genre_stats)
        
        return genre_df_expanded
    
    def analyze_actors(self):
        """Analiza los actores más populares."""
        credits_df = self.get_credits_dataframe()
        cast_df = credits_df[credits_df['credit_type'] == 'cast'].copy()
        
        # Estadísticas por actor
        actor_stats = cast_df.groupby('name').agg({
            'movie_id': 'count',
            'vote_average': 'mean',
            'popularity': 'mean'
        }).rename(columns={'movie_id': 'movie_count'})
        
        # Top 10 actores por número de películas
        top_actors = actor_stats.sort_values('movie_count', ascending=False).head(10)
        
        print("\n=== ANÁLISIS DE ACTORES ===")
        print("Top 10 actores por número de películas:")
        print(top_actors)
        
        return actor_stats
    
    def create_visualizations(self):
        """Crea visualizaciones de los datos."""
        df = self.get_movies_dataframe()
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución de calificaciones
        axes[0, 0].hist(df['vote_average'].dropna(), bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribución de Calificaciones')
        axes[0, 0].set_xlabel('Calificación Promedio')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # 2. Relación entre presupuesto y ingresos
        budget_revenue = df[(df['budget'] > 0) & (df['revenue'] > 0)]
        axes[0, 1].scatter(budget_revenue['budget'], budget_revenue['revenue'], alpha=0.6)
        axes[0, 1].set_title('Presupuesto vs Ingresos')
        axes[0, 1].set_xlabel('Presupuesto')
        axes[0, 1].set_ylabel('Ingresos')
        
        # 3. Películas por año
        df['year'] = df['release_date'].dt.year
        year_counts = df['year'].value_counts().sort_index()
        axes[1, 0].plot(year_counts.index, year_counts.values, marker='o')
        axes[1, 0].set_title('Películas por Año')
        axes[1, 0].set_xlabel('Año')
        axes[1, 0].set_ylabel('Número de Películas')
        
        # 4. Top 10 géneros por popularidad
        genre_df = self.analyze_genres()
        top_genres = genre_df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(top_genres)), top_genres.values)
        axes[1, 1].set_yticks(range(len(top_genres)))
        axes[1, 1].set_yticklabels(top_genres.index)
        axes[1, 1].set_title('Top 10 Géneros por Popularidad')
        axes[1, 1].set_xlabel('Popularidad Promedio')
        
        plt.tight_layout()
        plt.savefig('data/movie_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizaciones guardadas en 'data/movie_analysis.png'")
    
    def generate_report(self):
        """Genera un reporte completo de análisis."""
        print("Generando reporte de análisis...")
        
        # Análisis básico
        movies_df = self.analyze_movie_trends()
        genres_df = self.analyze_genres()
        actors_df = self.analyze_actors()
        
        # Crear visualizaciones
        self.create_visualizations()
        
        # Guardar datos procesados
        movies_df.to_csv('data/movies_analysis.csv', index=False)
        genres_df.to_csv('data/genres_analysis.csv', index=False)
        actors_df.to_csv('data/actors_analysis.csv', index=False)
        
        print("Reporte generado exitosamente!")
        print("Archivos guardados en el directorio 'data/'")
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        self.session.close()

def main():
    """Función principal para ejecutar el análisis."""
    from src.utils.logger import setup_logger
    
    # Configurar logging
    setup_logger()
    
    # Validar configuración
    if not Config.validate():
        logger.error("Configuración inválida. Revisa el archivo .env")
        return
    
    # Crear directorio de datos si no existe
    import os
    os.makedirs('data', exist_ok=True)
    
    # Ejecutar análisis
    analyzer = MovieDataAnalyzer()
    
    try:
        analyzer.generate_report()
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()
