import os
keys_to_check = [
    "AZURE_OPENAI_API_KEY", "AZURE_AI_API_KEY", 
    "AZURE_OPENAI_ENDPOINT", "AZURE_AI_ENDPOINT",
    "MYSQL_PASSWORD", "MYSQL_USER", "MYSQL_DATABASE"
]
for key in keys_to_check:
    val = os.getenv(key)
    if val:
        print(f"{key}: SET (length {len(val)})")
    else:
        print(f"{key}: NOT SET")
