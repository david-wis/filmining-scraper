"""
Database connection utilities for the Streamlit ROI prediction app.
"""
import os
from sqlalchemy import create_engine, text
import streamlit as st


def get_database_connection():
    """
    Create and return a database connection using environment variables or defaults.
    
    Returns:
        sqlalchemy.engine.Engine: Database engine
    """
    # Database configuration
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '25432')
    db_name = os.getenv('DB_NAME', 'movie_database')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    
    # Create connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        st.error("Please make sure the PostgreSQL database is running and accessible.")
        return None


@st.cache_data
def test_database_connection():
    """
    Test database connection and return status.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    engine = get_database_connection()
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM movies"))
            count = result.scalar()
            st.success(f"✅ Database connected successfully! Found {count:,} movies.")
            return True
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return False
