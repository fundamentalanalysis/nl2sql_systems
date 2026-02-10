from mcp_tools.get_schema import get_schema
import json
import time

print("Testing optimized get_schema...")
start_time = time.time()
try:
    schema = get_schema()
    end_time = time.time()
    
    print(f"Success! Retrieved schema in {end_time - start_time:.4f} seconds")
    print(f"Table count: {len(schema['tables'])}")
    if len(schema['tables']) > 0:
        print(f"First table: {schema['tables'][0]['name']}")
        print(f"Columns: {len(schema['tables'][0]['columns'])}")

except Exception as e:
    print(f"Error: {e}")
