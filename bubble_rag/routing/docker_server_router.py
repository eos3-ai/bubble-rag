from datetime import datetime

from fastapi import APIRouter
from sqlalchemy import func, delete as sqldelete
from sqlmodel import select, and_

from bubble_rag.databases.relation_database import SessionDep, update_model_from_dict
from bubble_rag.entity.relational.models import DockerServer
from bubble_rag.entity.query.models import DockerServerReq
from bubble_rag.entity.query.response_model import SrvResult, PageResult
from bubble_rag.utils.snowflake_utils import gen_id

router = APIRouter()


@router.post("/add_docker_server")
async def add_docker_server(server_param: DockerServerReq, session: SessionDep):
    """添加Docker服务器"""
    docker_server = DockerServer(**server_param.dict(exclude={'page_size', 'page_num', 'server_id'}))
    docker_server.id = gen_id()
    session.add(docker_server)
    session.commit()
    session.refresh(docker_server)
    return SrvResult(code=200, msg='success', data=docker_server)


@router.post("/edit_docker_server")
async def edit_docker_server(server_param: DockerServerReq, session: SessionDep):
    """编辑Docker服务器"""
    if server_param and server_param.id:
        db_server = session.get(DockerServer, server_param.id)
        if db_server:
            update_data = server_param.dict(exclude={'id', 'page_size', 'page_num', 'server_id'})
            # 过滤空值
            update_data = {k: v for k, v in update_data.items() if v is not None and v != ""}
            update_data['update_time'] = datetime.now()

            update_model_from_dict(db_server, update_data)
            session.commit()
            session.refresh(db_server)
            return SrvResult(code=200, msg='success', data=db_server)
        return SrvResult(code=404, msg='Docker服务器不存在', data=None)
    return SrvResult(code=400, msg='缺少必要参数', data=None)


@router.post("/delete_docker_server")
async def delete_docker_server(server_param: DockerServerReq, session: SessionDep):
    """删除Docker服务器"""
    server_id = server_param.server_id or server_param.id
    if server_id:
        # 检查是否存在
        db_server = session.get(DockerServer, server_id)
        if db_server:
            session.exec(sqldelete(DockerServer).where(DockerServer.id == server_id))
            session.commit()
            return SrvResult(code=200, msg='success', data=None)
        return SrvResult(code=404, msg='Docker服务器不存在', data=None)
    return SrvResult(code=400, msg='缺少服务器ID', data=None)


@router.post("/list_all_docker_servers")
async def list_docker_servers(server_param: DockerServerReq, session: SessionDep):
    """查询Docker服务器列表"""
    # 构建查询语句
    statement = select(DockerServer).order_by(DockerServer.update_time.desc())
    servers = session.exec(statement).all()
    return SrvResult(code=200, msg='success', data=servers)


@router.post("/list_docker_servers")
async def list_docker_servers(server_param: DockerServerReq, session: SessionDep):
    """查询Docker服务器列表"""
    conditions = []
    page_num = server_param.page_num
    if page_num < 1:
        page_num = 1
    page_size = server_param.page_size
    offset = (page_num - 1) * page_size

    # 添加搜索条件
    if server_param.server_name and server_param.server_name.strip():
        conditions.append(DockerServer.server_name.like(f"%{server_param.server_name}%"))
    if server_param.srv_base_url and server_param.srv_base_url.strip():
        conditions.append(DockerServer.srv_base_url.like(f"%{server_param.srv_base_url}%"))

    # 构建查询语句
    statement = select(DockerServer).where(and_(*conditions)).order_by(DockerServer.update_time.desc())
    statement_page = select(func.count()).where(and_(*conditions)).select_from(DockerServer)

    # 计算总数
    total = session.exec(statement_page).one()

    # 应用分页
    statement = statement.offset(offset).limit(page_size)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    servers = session.exec(statement).all()

    page_result = PageResult(items=servers, total=total, page=page_num, page_size=page_size, total_pages=total_pages)
    return SrvResult(code=200, msg='success', data=page_result)


@router.post("/get_docker_server")
async def get_docker_server(server_param: DockerServerReq, session: SessionDep):
    """获取Docker服务器详情"""
    server_id = server_param.server_id or server_param.id
    if server_id:
        db_server = session.get(DockerServer, server_id)
        if db_server:
            return SrvResult(code=200, msg='success', data=db_server)
        return SrvResult(code=404, msg='Docker服务器不存在', data=None)
    return SrvResult(code=400, msg='缺少服务器ID', data=None)


@router.post("/check_docker_server_status")
async def check_docker_server_status(server_param: DockerServerReq, session: SessionDep):
    """检查Docker服务器状态"""
    server_id = server_param.server_id or server_param.id
    if server_id:
        db_server = session.get(DockerServer, server_id)
        if db_server:
            try:
                import requests
                # 简单的健康检查，可以根据实际需求调整
                response = requests.get(f"{db_server.srv_base_url.rstrip('/')}/health", timeout=5)
                if response.status_code == 200:
                    status = {"status": "online", "message": "服务器在线"}
                else:
                    status = {"status": "offline", "message": f"服务器响应异常: {response.status_code}"}
            except Exception as e:
                status = {"status": "offline", "message": f"连接失败: {str(e)}"}

            return SrvResult(code=200, msg='success', data={"server": db_server, "status": status})
        return SrvResult(code=404, msg='Docker服务器不存在', data=None)
    return SrvResult(code=400, msg='缺少服务器ID', data=None)
