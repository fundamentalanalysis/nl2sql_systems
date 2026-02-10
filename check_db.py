import pymysql
try:
    conn = pymysql.connect(host='localhost', user='root', password='')
    with conn.cursor() as cursor:
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        print(f"Databases: {databases}")
except Exception as e:
    print(f"Error: {e}")
