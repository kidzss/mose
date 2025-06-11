<<<<<<< HEAD
import pymysql
import sys

def test_mysql_connection():
    print("Testing MySQL connection...")
    try:
        # 尝试不同的连接方式
        connection_params = [
            {"host": "localhost", "user": "root", "password": "123456"},
            {"host": "127.0.0.1", "user": "root", "password": "123456"},
            {"host": "localhost", "user": "root", "password": ""},
            {"host": "127.0.0.1", "user": "root", "password": ""}
        ]
        
        for params in connection_params:
            print(f"\nTrying connection with: {params}")
            try:
                conn = pymysql.connect(**params)
                print("Connection successful!")
                cursor = conn.cursor()
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                print(f"MySQL version: {version[0]}")
                cursor.execute("SHOW DATABASES")
                databases = cursor.fetchall()
                print("Available databases:")
                for db in databases:
                    print(f"- {db[0]}")
                conn.close()
                return True
            except pymysql.Error as e:
                print(f"Connection failed: {e}")
                continue
                
        print("\nAll connection attempts failed.")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_mysql_connection():
        print("\nMySQL connection test completed successfully!")
    else:
        print("\nMySQL connection test failed!")
=======
import pymysql
import sys

def test_mysql_connection():
    print("Testing MySQL connection...")
    try:
        # 尝试不同的连接方式
        connection_params = [
            {"host": "localhost", "user": "root", "password": "123456"},
            {"host": "127.0.0.1", "user": "root", "password": "123456"},
            {"host": "localhost", "user": "root", "password": ""},
            {"host": "127.0.0.1", "user": "root", "password": ""}
        ]
        
        for params in connection_params:
            print(f"\nTrying connection with: {params}")
            try:
                conn = pymysql.connect(**params)
                print("Connection successful!")
                cursor = conn.cursor()
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                print(f"MySQL version: {version[0]}")
                cursor.execute("SHOW DATABASES")
                databases = cursor.fetchall()
                print("Available databases:")
                for db in databases:
                    print(f"- {db[0]}")
                conn.close()
                return True
            except pymysql.Error as e:
                print(f"Connection failed: {e}")
                continue
                
        print("\nAll connection attempts failed.")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_mysql_connection():
        print("\nMySQL connection test completed successfully!")
    else:
        print("\nMySQL connection test failed!")
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
        sys.exit(1) 