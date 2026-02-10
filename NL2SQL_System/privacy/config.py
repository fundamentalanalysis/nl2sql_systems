from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv

load_dotenv()

# Key should be persistent for a session to allow decoding
# In a real app, this would be a static key from env or a KMS
_fernet_key = os.getenv("PII_ENCRYPTION_KEY")
if not _fernet_key:
    # Generate a temporary key if none provided - ONLY for demo purposes
    # In production, this would lead to data loss if server restarts
    _fernet_key = Fernet.generate_key().decode()

fernet = Fernet(_fernet_key.encode())

def encrypt_value(value: str) -> str:
    """Encrypt a string value using Fernet."""
    return fernet.encrypt(value.encode()).decode()

def decrypt_value(token: str) -> str:
    """Decrypt a Fernet token back to string."""
    return fernet.decrypt(token.encode()).decode()
