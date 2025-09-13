from sonyflake import SonyFlake

from sonyflake import SonyFlake
from datetime import datetime, timezone
import socket


# 自定义机器ID生成逻辑
def generate_machine_id() -> int:
    """通过IP地址最后两位生成机器ID"""
    host_ip = socket.gethostbyname(socket.gethostname())
    last_two_octets = list(map(int, host_ip.split('.')))[-2:]
    return (last_two_octets[0] << 8) | last_two_octets[1]


# 配置 Sonyflake
sf = SonyFlake(
    machine_id=generate_machine_id,
    start_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
    check_machine_id=lambda x: 0 <= x < 65536
)


# 生成ID
def gen_id():
    """生成全局唯一的Snowflake ID字符串"""
    id = sf.next_id()
    return str(id)
