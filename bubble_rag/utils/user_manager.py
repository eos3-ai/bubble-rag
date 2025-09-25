"""
简化的用户管理器
通过请求参数username进行用户身份识别，无需复杂认证
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from sqlmodel import Session, select

logger = logging.getLogger(__name__)


class UserManager:
    """简化的用户管理器 - 适用于开源项目"""

    @staticmethod
    def _get_database_session():
        """获取数据库会话"""
        try:
            from bubble_rag.training.mysql_service.entity.training_task_models import get_engine
            return Session(get_engine())
        except Exception as e:
            logger.error(f"无法获取数据库会话: {e}")
            return None

    @staticmethod
    def validate_and_get_user(username: Optional[str] = None) -> Dict[str, Any]:
        """
        验证并获取用户信息

        Args:
            username: 请求参数中的用户名，可选

        Returns:
            Dict: 用户信息字典

        Raises:
            HTTPException: 当指定的用户不存在时
        """
        # 1. 如果没有传username，默认使用admin
        if not username:
            username = 'admin'
            logger.debug("未指定用户名，使用默认admin用户")

        # 2. 检查用户是否在数据库中存在
        session = UserManager._get_database_session()
        if not session:
            # 数据库连接失败时的降级处理
            logger.warning("数据库连接失败，使用降级模式")
            return {
                'username': username,
                'user_role': 'admin' if username == 'admin' else 'user',
                'display_name': None,
                'is_admin': username == 'admin',
                'source': 'fallback'
            }

        try:
            from bubble_rag.training.mysql_service.entity.user_models import UserDB
            user = session.exec(select(UserDB).where(UserDB.username == username)).first()

            if user:
                # 用户存在，返回用户信息
                user_info = {
                    'username': user.username,
                    'user_role': user.user_role,
                    'display_name': user.display_name,
                    'is_admin': user.user_role == 'admin',
                    'source': 'database'
                }
                logger.debug(f"找到用户: {username}, 角色: {user.user_role}")
                return user_info
            else:
                # 用户不存在，抛出异常
                logger.warning(f"用户 {username} 不存在")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"用户 '{username}' 不存在，请先通过用户管理接口创建用户"
                )

        except HTTPException:
            # 重新抛出HTTP异常
            raise
        except Exception as e:
            logger.error(f"查询用户信息失败: {e}")
            # 数据库查询失败时的异常处理
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="用户验证服务异常，请稍后重试"
            )
        finally:
            session.close()

    @staticmethod
    def check_user_exists(username: str) -> bool:
        """
        检查用户是否存在（用于用户管理接口）

        Args:
            username: 要检查的用户名

        Returns:
            bool: 用户是否存在
        """
        session = UserManager._get_database_session()
        if not session:
            return False

        try:
            from bubble_rag.training.mysql_service.entity.user_models import UserDB
            user = session.exec(select(UserDB).where(UserDB.username == username)).first()
            return user is not None
        except Exception as e:
            logger.error(f"检查用户存在性失败: {e}")
            return False
        finally:
            session.close()

    @staticmethod
    def get_user_summary() -> Dict[str, Any]:
        """
        获取用户统计信息（用于管理界面）

        Returns:
            Dict: 用户统计信息
        """
        session = UserManager._get_database_session()
        if not session:
            return {"total_users": 0, "admin_users": 0, "regular_users": 0}

        try:
            from bubble_rag.training.mysql_service.entity.user_models import UserDB

            # 统计用户数量
            all_users = session.exec(select(UserDB)).all()
            total_users = len(all_users)
            admin_users = len([u for u in all_users if u.user_role == 'admin'])
            regular_users = total_users - admin_users

            return {
                "total_users": total_users,
                "admin_users": admin_users,
                "regular_users": regular_users,
                "users": [
                    {
                        "username": user.username,
                        "user_role": user.user_role,
                        "display_name": user.display_name,
                        "created_at": user.created_at.isoformat() if user.created_at else None
                    }
                    for user in all_users
                ]
            }
        except Exception as e:
            logger.error(f"获取用户统计失败: {e}")
            return {"error": str(e)}
        finally:
            session.close()

    @staticmethod
    def _can_access_task(task_username: str, current_user: Dict[str, Any] = None) -> bool:
        """
        检查是否可以访问指定任务（兼容性方法）

        Args:
            task_username: 任务的创建者用户名
            current_user: 当前用户信息，如果不传则获取默认用户

        Returns:
            bool: 是否有权限访问
        """
        if current_user is None:
            current_user = UserManager.validate_and_get_user()

        # 管理员可以访问所有任务
        if current_user.get('is_admin', False):
            return True

        # 普通用户只能访问自己的任务
        return task_username == current_user.get('username')


# 便捷函数
def validate_user(username: Optional[str] = None) -> Dict[str, Any]:
    """验证用户的便捷函数"""
    return UserManager.validate_and_get_user(username)