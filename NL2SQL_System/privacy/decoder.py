import re
from privacy.config import decrypt_value
from privacy.encoder import _token_store
from typing import Any, Dict

def decode_text(text: str) -> str:
    """
    Finds PII tokens in text and replaces them with decrypted original values.
    Pattern: [ENTITY_TYPE_HASH]
    """
    decoded_text = text
    
    # Regex to find tokens like [PERSON_A1B2C3D4]
    token_pattern = r"\[[A-Z_]+_[0-9A-F]{8}\]"
    tokens_found = re.findall(token_pattern, text)
    
    # Sort by length descending to avoid partial matching if tokens varied in length
    # (though they are fixed 8-char hashes here)
    for token in set(tokens_found):
        if token in _token_store:
            try:
                encrypted_val = _token_store[token]
                original_val = decrypt_value(encrypted_val)
                decoded_text = decoded_text.replace(token, original_val)
            except Exception as e:
                # If decryption fails for some reason, keep the token or mark error
                print(f"Error decoding token {token}: {e}")
                
    return decoded_text


def decode_results(result: Any) -> Any:
    """
    Decode PII tokens in structured query results.

    Supports:
    - {"columns": [...], "rows": [[...], ...], "row_count": N}
    - List of dicts or list of lists
    """
    if result is None:
        return result

    # Dict with columns/rows
    if isinstance(result, dict) and "rows" in result:
        rows = result.get("rows", [])
        decoded_rows = []
        for row in rows:
            decoded_row = []
            for val in row:
                if isinstance(val, str):
                    decoded_row.append(decode_text(val))
                else:
                    decoded_row.append(val)
            decoded_rows.append(decoded_row)
        decoded = dict(result)
        decoded["rows"] = decoded_rows
        return decoded

    # List of dicts
    if isinstance(result, list) and result and isinstance(result[0], dict):
        decoded_list = []
        for item in result:
            decoded_item = {}
            for k, v in item.items():
                decoded_item[k] = decode_text(v) if isinstance(v, str) else v
            decoded_list.append(decoded_item)
        return decoded_list

    # List of lists
    if isinstance(result, list) and result and isinstance(result[0], list):
        decoded_list = []
        for row in result:
            decoded_row = []
            for val in row:
                decoded_row.append(decode_text(val) if isinstance(val, str) else val)
            decoded_list.append(decoded_row)
        return decoded_list

    return result
