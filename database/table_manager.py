from database.config import *
import pymysql

class BaseTableManager:
    def __init__(self):
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=DB_HOST,
                port=int(DB_PORT),
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                charset='utf8',
                read_timeout=60 # with the read_timeout parameter being set the connection error is being thrown out
            )
            self.cursor = self.connection.cursor()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
    
    def close(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()
    
class UserTableManager(BaseTableManager):
    def __init__(self):
        super().__init__()
    
    def check_user(self, username, email):
        self.connect()

        sql = """
        SELECT * FROM user
        WHERE username = %s AND email = %s
        """
        values = (username, email)
        try:
            self.cursor.execute(sql, values)
            results = self.cursor.fetchone()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
        finally:
            self.close()
            return results
    
    def update_last_login(self, user_id):
        self.connect()

        sql = """
        UPDATE user
        SET last_login = CURRENT_TIMESTAMP
        WHERE user_id = %s
        """
        try:
            self.cursor.execute(sql, user_id)
            self.connection.commit()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
        finally:
            self.close()
    
    def create_user(self, username, email):
        self.connect()

        sql = """
        INSERT INTO user (username, email)
        VALUES (%s, %s)
        """
        values = (username, email)
        try:
            self.cursor.execute(sql, values)
            self.connection.commit()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
        finally:
            self.close()

class ChatLogTableManager(BaseTableManager):
    def __init__(self):
        super().__init__()

    def create_chat_title(self, session_id, user_id, chat_title):
        self.connect()

        sql = """
        INSERT INTO chat_title (session_id, user_id, title)
        VALUES (%s, %s, %s)
        """
        values = (session_id, user_id, chat_title)
        try:
            self.cursor.execute(sql, values)
            self.connection.commit()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
        finally:
            self.close()
    
    def insert_chat_log(self, session_id, user_id, sender, message):
        self.connect()

        sql = """
        INSERT INTO chat_log (session_id, user_id, sender, message)
        VALUES (%s, %s, %s, %s)
        """
        values = (session_id, user_id, sender, message)
        try:
            self.cursor.execute(sql, values)
            self.connection.commit()
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
        finally:
            self.close()