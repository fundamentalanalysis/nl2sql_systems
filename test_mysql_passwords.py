import pymysql
import sys

COMMON_PASSWORDS = ["", "root", "password", "admin", "123456"]
HOST = "localhost"
USER = "root"

print(f"Testing MySQL connection for user '{USER}' on '{HOST}' with common passwords...\n")

success = False
for pwd in COMMON_PASSWORDS:
    try:
        print(f"Trying password: '{pwd}' ... ", end="")
        conn = pymysql.connect(host=HOST, user=USER, password=pwd)
        print("SUCCESS! ✅")
        print(f"\nYour correct password is: '{pwd}'")
        print("Please update your .env file with this password.")
        conn.close()
        success = True
        break
    except pymysql.err.OperationalError as e:
        if e.args[0] == 1045: # Access denied
            print("Access Denied ❌")
        else:
            print(f"Error ({e.args[0]}): {e.args[1]}")
    except Exception as e:
        print(f"Connection Error: {e}")

if not success:
    print("\n❌ None of the common passwords worked.")
    print("Please check your MySQL configuration manually or reset the root password.")
