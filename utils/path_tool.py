from pathlib import Path

def get_root_path():
    """获取项目根目录的绝对路径"""
    return Path(__file__).parent.parent.resolve()

def get_abs_path(relative_path):
    """将相对路径转换为基于项目根目录的绝对路径"""
    return get_root_path() / relative_path