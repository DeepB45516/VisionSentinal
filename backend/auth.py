import sqlite3

DB = "database.db"


def create_user(username, password):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    try:

        cursor.execute(
            "INSERT INTO users(username,password) VALUES (?,?)",
            (username, password)
        )

        conn.commit()
        conn.close()

        return True

    except:

        conn.close()
        return False


def verify_user(username, password):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()

    conn.close()

    return user is not None
def create_default_user():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    try:
        cursor.execute(
            "INSERT INTO users(username,password) VALUES (?,?)",
            ("admin","admin123")
        )
        conn.commit()
    except:
        pass

    conn.close()