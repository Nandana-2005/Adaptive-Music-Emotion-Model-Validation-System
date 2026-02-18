import sqlite3
from datetime import datetime

def create_tables():
    """Create all database tables"""
    conn = sqlite3.connect('amecs.db')
    cursor = conn.cursor()
    
    # Table 1: Children/Users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table 2: Music Interactions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            emotion TEXT NOT NULL,
            music_file TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Table 3: Caregiver Feedback
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER,
            accurate BOOLEAN,
            notes TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Tables created successfully!")


def insert_user(name, age):
    """Add a new child to database"""
    conn = sqlite3.connect('amecs.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', (name, age))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

def log_interaction(user_id, emotion, music_file, session_id):
    """Log when child selects music"""
    conn = sqlite3.connect('amecs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO interactions (user_id, emotion, music_file, session_id)
        VALUES (?, ?, ?, ?)
    ''', (user_id, emotion, music_file, session_id))
    interaction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return interaction_id

def get_user_history(user_id):
    """Get all interactions for a child"""
    conn = sqlite3.connect('amecs.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT emotion, music_file, timestamp 
        FROM interactions 
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history

def save_feedback(interaction_id, accurate, notes=""):
    """Save caregiver feedback"""
    conn = sqlite3.connect('amecs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (interaction_id, accurate, notes)
        VALUES (?, ?, ?)
    ''', (interaction_id, accurate, notes))
    conn.commit()
    conn.close()



if __name__ == "__main__":
    create_tables()