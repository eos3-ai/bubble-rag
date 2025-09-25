"""用户管理API路由"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from sqlmodel import Session, select
from pydantic import BaseModel

from bubble_rag.training.mysql_service.entity.user_models import UserDB, UserCreate, UserResponse, UserUpdate
from bubble_rag.training.mysql_service.entity.training_task_models import get_engine
from bubble_rag.utils.user_manager import UserManager

router = APIRouter()


def get_session():
    """获取数据库会话"""
    engine = get_engine()
    with Session(engine) as session:
        yield session


class AuthRequest(BaseModel):
    """用户认证请求模型"""
    username: str
    password: str


class AuthResponse(BaseModel):
    """用户认证响应模型"""
    username: str
    user_role: str
    display_name: Optional[str]
    access_token: str  # 简化版本，实际应该使用JWT
    message: str


@router.post("/auth/login", response_model=AuthResponse, summary="用户登录")
async def login(auth_request: AuthRequest, session: Session = Depends(get_session)):
    """用户登录认证"""
    # 使用UserManager验证用户存在性，然后手动验证密码
    import hashlib

    # 首先验证用户是否存在
    if not UserManager.check_user_exists(auth_request.username):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

    # 验证密码
    user = session.exec(select(UserDB).where(UserDB.username == auth_request.username)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

    password_hash = hashlib.sha256(auth_request.password.encode()).hexdigest()
    if user.user_password != password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

    user_info = {
        'username': user.username,
        'user_role': user.user_role,
        'display_name': user.display_name
    }

    # 简化版token生成（实际应该使用JWT）
    access_token = f"token_{user_info['username']}_{hashlib.md5(auth_request.password.encode()).hexdigest()[:8]}"

    return AuthResponse(
        username=user_info['username'],
        user_role=user_info['user_role'],
        display_name=user_info.get('display_name'),
        access_token=access_token,
        message="登录成功"
    )


@router.get("/list", response_model=List[UserResponse], summary="获取用户列表")
async def get_users(session: Session = Depends(get_session)):
    """获取所有用户列表"""
    users = session.exec(select(UserDB)).all()
    return [
        UserResponse(
            username=user.username,
            user_role=user.user_role,
            display_name=user.display_name,
            created_at=user.created_at,
            updated_at=user.updated_at
        ) for user in users
    ]


@router.get("/{username}", response_model=UserResponse, summary="获取指定用户信息")
async def get_user(username: str, session: Session = Depends(get_session)):
    """根据用户名获取用户信息"""
    user = session.exec(select(UserDB).where(UserDB.username == username)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {username} 不存在"
        )

    return UserResponse(
        username=user.username,
        user_role=user.user_role,
        display_name=user.display_name,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.post("/register", response_model=UserResponse, summary="创建新用户")
async def create_user(user_create: UserCreate, session: Session = Depends(get_session)):
    """创建新用户"""
    # 检查用户是否已存在
    existing_user = session.exec(select(UserDB).where(UserDB.username == user_create.username)).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"用户名 {user_create.username} 已存在"
        )

    # 创建新用户
    password_hash = hashlib.sha256(user_create.user_password.encode()).hexdigest()

    new_user = UserDB(
        username=user_create.username,
        user_password=password_hash,
        user_role=user_create.user_role,
        display_name=user_create.display_name
    )

    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return UserResponse(
        username=new_user.username,
        user_role=new_user.user_role,
        display_name=new_user.display_name,
        created_at=new_user.created_at,
        updated_at=new_user.updated_at
    )


@router.put("/{username}", response_model=UserResponse, summary="更新用户信息")
async def update_user(username: str, user_update: UserUpdate, session: Session = Depends(get_session)):
    """更新用户信息"""
    user = session.exec(select(UserDB).where(UserDB.username == username)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {username} 不存在"
        )

    # 更新用户信息
    if user_update.user_password is not None:
        user.user_password = hashlib.sha256(user_update.user_password.encode()).hexdigest()

    if user_update.user_role is not None:
        user.user_role = user_update.user_role

    if user_update.display_name is not None:
        user.display_name = user_update.display_name

    session.add(user)
    session.commit()
    session.refresh(user)

    return UserResponse(
        username=user.username,
        user_role=user.user_role,
        display_name=user.display_name,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.delete("/{username}", summary="删除用户")
async def delete_user(username: str, session: Session = Depends(get_session)):
    """删除用户"""
    # 防止删除admin用户
    if username == 'admin':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除admin管理员用户"
        )

    user = session.exec(select(UserDB).where(UserDB.username == username)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {username} 不存在"
        )

    session.delete(user)
    session.commit()

    return {"message": f"用户 {username} 删除成功"}


@router.post("/{username}/reset-password", summary="重置用户密码")
async def reset_password(
    username: str,
    new_password: str,
    session: Session = Depends(get_session)
):
    """重置用户密码"""
    user = session.exec(select(UserDB).where(UserDB.username == username)).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {username} 不存在"
        )

    # 更新密码
    user.user_password = hashlib.sha256(new_password.encode()).hexdigest()
    session.add(user)
    session.commit()

    return {"message": f"用户 {username} 密码重置成功"}


@router.post("/init-admin", summary="初始化默认管理员用户")
async def init_admin_user(session: Session = Depends(get_session)):
    """手动初始化默认管理员用户（用于开发调试）"""
    # 检查admin用户是否已存在
    admin_user = session.exec(select(UserDB).where(UserDB.username == 'admin')).first()

    if admin_user:
        return {"message": "admin用户已存在", "username": "admin", "role": admin_user.user_role}

    # 创建默认admin用户
    default_admin = UserDB(
        username='admin',
        user_password=hashlib.sha256('admin'.encode()).hexdigest(),
        user_role='admin',
        display_name='系统管理员'
    )

    session.add(default_admin)
    session.commit()

    return {
        "message": "默认admin用户创建成功",
        "username": "admin",
        "password": "admin",
        "role": "admin"
    }