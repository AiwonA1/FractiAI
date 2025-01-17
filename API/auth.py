"""
Authentication module for FractiAI API

Implements JWT-based authentication with role-based access control
and sophisticated security features.
"""

from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

# Security configuration
SECRET_KEY = "your-secret-key"  # Should be loaded from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize security components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class User(BaseModel):
    """User model with role-based permissions"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole
    disabled: bool = False
    permissions: List[str] = []

class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[str] = []

# Role-based permissions
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        "simulation:create",
        "simulation:delete",
        "simulation:modify",
        "analysis:all",
        "system:configure"
    ],
    UserRole.RESEARCHER: [
        "simulation:create",
        "simulation:modify",
        "analysis:basic",
        "analysis:advanced"
    ],
    UserRole.VIEWER: [
        "simulation:view",
        "analysis:basic"
    ]
}

class AuthManager:
    """Manages authentication and authorization"""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._tokens: Dict[str, datetime] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
        
    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """Authenticate user credentials"""
        user = self._users.get(username)
        if not user:
            return None
            
        if not self.verify_password(password, user.hashed_password):
            return None
            
        return user
        
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
            
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        self._tokens[encoded_jwt] = expire
        
        return encoded_jwt
        
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
    async def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            
            if username is None or role is None:
                return None
                
            return TokenData(
                username=username,
                role=UserRole(role),
                permissions=ROLE_PERMISSIONS[UserRole(role)]
            )
            
        except JWTError:
            return None
            
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
        
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
        
    async def check_permission(
        self,
        token_data: TokenData,
        required_permission: str
    ) -> bool:
        """Check if user has required permission"""
        return required_permission in token_data.permissions
        
    async def revoke_token(self, token: str) -> bool:
        """Revoke an active token"""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False
        
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens"""
        current_time = datetime.utcnow()
        expired = [
            token for token, expire in self._tokens.items()
            if expire < current_time
        ]
        
        for token in expired:
            del self._tokens[token]
            
    async def check_rate_limit(
        self,
        username: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """Check rate limiting for user"""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=window)
        
        # Clean up old requests
        if username in self._rate_limits:
            self._rate_limits[username] = [
                t for t in self._rate_limits[username]
                if t > window_start
            ]
        else:
            self._rate_limits[username] = []
            
        # Check limit
        if len(self._rate_limits[username]) >= limit:
            return False
            
        # Add new request
        self._rate_limits[username].append(current_time)
        return True

# Initialize auth manager
auth_manager = AuthManager()

async def get_current_user(
    token: str = Security(oauth2_scheme)
) -> User:
    """Get current user from token"""
    token_data = await auth_manager.verify_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
        
    user = auth_manager._users.get(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
        
    if user.disabled:
        raise HTTPException(
            status_code=400,
            detail="User is disabled"
        )
        
    return user

async def get_current_active_user(
    current_user: User = Security(get_current_user)
) -> User:
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(
            status_code=400,
            detail="Inactive user"
        )
    return current_user

# Token creation helper
async def create_token_response(user: User) -> Token:
    """Create token response for user"""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token = auth_manager.create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "permissions": user.permissions
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = auth_manager.create_refresh_token(
        data={"sub": user.username}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token
    )

# Start background token cleanup task
async def cleanup_task():
    """Background task for cleaning up expired tokens"""
    while True:
        await auth_manager.cleanup_expired_tokens()
        await asyncio.sleep(300)  # Run every 5 minutes

# Initialize cleanup task
asyncio.create_task(cleanup_task()) 