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
            return None
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
            if isinstance(e, pymysql.err.IntegrityError):
                return "Duplicate entry"
            else:
                return f"Unknown error {e}"
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

    def get_new_session_id(self, user_id):
        # 사용자의 마지막 session_id를 가져와 1을 더해 새로운 session_id 반환
        self.connect()
        sql = """
        SELECT MAX(session_id) AS last_session
        FROM chat_title
        WHERE user_id = %s
        """
        try:
            self.cursor.execute(sql, (user_id,))
            result = self.cursor.fetchone()
            last_session_id = result[0] if result[0] else 0
            return last_session_id + 1
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
            return 9999
        finally:
            self.close()
    
    def get_chat_titles(self, user_id):
        self.connect()
        sql = """
        SELECT * FROM chat_title
        WHERE user_id = %s
        """
        try:
            self.cursor.execute(sql, (user_id,))
            result = self.cursor.fetchall()
            return result
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
            return 9999
        finally:
            self.close()
    
    def get_session_chat(self, user_id, session_id):
        self.connect()
        sql = """
        SELECT * FROM chat_log
        WHERE user_id = %s AND session_id = %s
        """
        try:
            self.cursor.execute(sql, (user_id, session_id))
            result = self.cursor.fetchall()
            return result
        except pymysql.MySQLError as e:
            print(f">>> MySQL Error: {e}")
            return 9999
        finally:
            self.close()