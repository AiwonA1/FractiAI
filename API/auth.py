"""
auth.py

Authentication and authorization system for FractiAI API.
"""

from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from typing import Optional, Dict
import jwt
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration
SECRET_KEY = "your-secret-key"  # Should be in environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthManager:
    """Manages authentication and authorization"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.user_permissions: Dict[str, Dict] = {}
        
    async def verify_api_key(self, api_key: str = Security(api_key_header)) -> bool:
        """Verify API key"""
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
            
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
            
        return True
        
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
    async def verify_token(self, token: str = Depends(oauth2_scheme)) -> Dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
            
    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check user permissions"""
        if user_id not in self.user_permissions:
            return False
            
        permissions = self.user_permissions[user_id]
        return permissions.get(f"{resource}:{action}", False)
        
    def add_api_key(self, api_key: str, metadata: Dict) -> None:
        """Add new API key"""
        self.api_keys[api_key] = {
            "created_at": datetime.utcnow(),
            "metadata": metadata
        }
        
    def revoke_api_key(self, api_key: str) -> None:
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            
    def set_user_permissions(self, user_id: str, permissions: Dict) -> None:
        """Set user permissions"""
        self.user_permissions[user_id] = permissions

# Initialize auth manager
auth_manager = AuthManager() 