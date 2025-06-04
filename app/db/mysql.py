import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import logging
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection(use_database=True):
    try:
        connection_params = {
            'host': os.getenv("DB_HOST", "localhost"),
            'user': os.getenv("DB_USER", "root"),
            'password': os.getenv("DB_PASSWORD", "Saish@05"),
        }
        
        if use_database:
            connection_params['database'] = os.getenv("DB_NAME", "webchat_db")
            
        logger.info(f"Attempting to connect to MySQL {'database' if use_database else 'server'}...")
        connection = mysql.connector.connect(**connection_params)
        logger.info(f"Successfully connected to MySQL {'database' if use_database else 'server'}")
        return connection
    except Error as e:
        logger.error(f"Failed to connect to MySQL: {e}")
        raise e

def create_database():
    """Create database if it doesn't exist"""
    connection = None
    cursor = None
    try:
        logger.info("Starting database creation process...")
        connection = get_db_connection(use_database=False)
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        db_name = os.getenv("DB_NAME", "webchat_db")
        logger.info(f"Creating database '{db_name}' if it doesn't exist...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        connection.commit()
        logger.info(f"Database '{db_name}' created or already exists")
        
    except Error as e:
        logger.error(f"Failed to create database: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

def init_db():
    """Initialize database and create tables if they don't exist"""
    connection = None
    cursor = None
    try:
        # First ensure database exists
        logger.info("Starting database initialization...")
        create_database()
        
        # Then connect to the database and create tables
        connection = get_db_connection()
        cursor = connection.cursor()

        # Create users table
        logger.info("Creating users table if not exists...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        connection.commit()
        logger.info("Users table created or already exists")
        
        # Verify table creation
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        logger.info(f"Current tables in database: {[table[0] for table in tables]}")
        
        # Verify users table structure
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        logger.info("Users table structure:")
        for column in columns:
            logger.info(f"Column: {column[0]}, Type: {column[1]}, Null: {column[2]}, Key: {column[3]}, Default: {column[4]}")
            
    except Error as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

def get_db():
    """Get database connection"""
    connection = get_db_connection()
    try:
        yield connection
    finally:
        if connection.is_connected():
            connection.close()
            logger.info("Database connection closed") 