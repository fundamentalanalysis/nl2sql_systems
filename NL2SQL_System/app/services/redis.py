"""
Redis service wrapper for session management.
"""
import redis
from loguru import logger
from app.config import settings


class RedisService:
    """Redis client wrapper for token storage and session management."""

    def __init__(self):
        self.connected = False
        try:
            logger.info(f"Attempting to connect to Redis at {settings.redis_host}:{settings.redis_port}")
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                username=settings.redis_user or None,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
            self.connected = False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.connected

    def set_token(self, user_id: str, token: str, expiry_seconds: int = 86400) -> bool:
        """Store a token in Redis with expiration."""
        if not self.client:
            logger.warning("Redis not connected, skipping token storage")
            return False
            
        try:
            # Key: "token:{token}" -> Value: user_id
            key = f"token:{token}"
            self.client.setex(key, expiry_seconds, user_id)
            return True
        except Exception as e:
            logger.error(f"Error storing token in Redis: {e}")
            return False

    def get_token_user(self, token: str) -> str:
        """Retrieve user_id for a given token."""
        if not self.client:
            # If Redis is down, we might want to fail open or closed.
            # For strict session management, we should fail closed (return None).
            return None
            
        try:
            key = f"token:{token}"
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Error retrieving token from Redis: {e}")
            return None

    def delete_token(self, token: str) -> bool:
        """Remove a token from Redis (logout)."""
        if not self.client:
            return False
            
        try:
            key = f"token:{token}"
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting token from Redis: {e}")
            return False


# Global Redis instance
redis_client = RedisService()
