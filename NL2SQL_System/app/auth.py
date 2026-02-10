"""JWT authentication and authorization utilities."""
from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from loguru import logger
from app.config import settings
# Import Redis service
from app.services.redis import redis_client

# HTTP Bearer security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency to validate JWT and extract user context.

    Args:
        credentials: HTTP Authorization header credentials

    Returns:
        Dict containing user_id and role

    Raises:
        HTTPException: If token is invalid, expired, or role is invalid
    """
    token = credentials.credentials

    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        # print("DEBUG: JWT token decoded successfully", flush=True)

    except jwt.ExpiredSignatureError:
        print("DEBUG: JWT token expired", flush=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed token",
        )
    except Exception as e:
        logger.error(f"Unexpected error decoding JWT: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )

    # Extract user information
    user_id = payload.get("sub")
    role = payload.get("role")

    # Validate role
    if role not in ("admin", "viewer"):
        logger.warning(f"Invalid role in token: {role}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid user role",
        )

    if not user_id:
        logger.warning("Missing user_id in token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID missing from token",
        )

    # ---------------------------------------------------------
    # REDIS CHECK: Validate token against Redis (allowlist/session)
    # ---------------------------------------------------------
    if redis_client.is_connected:
        redis_valid = redis_client.get_token_user(token)
        if not redis_valid:
            print(f"DEBUG: Token not found in Redis (or expired): {user_id}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid (Redis check failed)",
            )
        
        # Optional: Verify the user_id in Redis matches the token's user_id
        if isinstance(redis_valid, bytes):
            redis_valid = redis_valid.decode('utf-8')
            
        if str(redis_valid) != str(user_id):
            logger.warning(f"Token user mismatch. Token: {user_id}, Redis: {redis_valid}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session validation mismatch",
            )
    else:
        logger.warning("Redis is not connected; skipping session validation (falling back to stateless JWT)")
    # ---------------------------------------------------------

    user_context = {
        "user_id": user_id,
        "role": role,
    }

    logger.debug(f"Authenticated user {user_id} with role {role}")
    return user_context


def create_jwt_token(user_id: str, role: str) -> str:
    """
    Create a JWT token for a user and store it in Redis.

    Args:
        user_id: Unique user identifier
        role: User role ('admin' or 'viewer')

    Returns:
        Signed JWT token string
    """
    import datetime

    # 24 hour expiration
    expiry_seconds = 24 * 3600
    
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expiry_seconds),
        "iat": datetime.datetime.now(datetime.timezone.utc),
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    
    # Store in Redis
    if isinstance(token, bytes):
        token_str = token.decode('utf-8')
    else:
        token_str = token
        
    stored = redis_client.set_token(user_id, token_str, expiry_seconds=expiry_seconds)
    if stored:
        logger.info(f"Token stored in Redis for user {user_id}")
    else:
        logger.warning(f"Failed to store token in Redis for user {user_id}")

    return token_str
