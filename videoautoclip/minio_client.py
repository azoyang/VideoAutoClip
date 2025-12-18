# -*- coding: utf-8 -*-
"""
MinIO客户端配置和工具函数
"""
import json
import os
import re
import uuid
from io import BytesIO
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error
from typing import Optional

def _bool_env(val: Optional[str]) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def get_minio_config():
    endpoint = os.environ.get("MINIO_ENDPOINT") or os.environ.get("MINIO_ENDPOINT_URL") or "47.80.6.218:9000"
    access_key = os.environ.get("MINIO_ACCESS_KEY") or "minioadmin"
    secret_key = os.environ.get("MINIO_SECRET_KEY") or "minioclip4123"
    secure = _bool_env(os.environ.get("MINIO_SECURE", "false"))
    bucket_name = os.environ.get("MINIO_BUCKET_NAME", "audio-files")
    base_url = os.environ.get("MINIO_BASE_URL") or (f"http://{endpoint}" if endpoint and not secure else f"https://{endpoint}" if endpoint else "")
    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "secure": secure,
        "bucket_name": bucket_name,
        "base_url": base_url,
    }

MINIO_CONFIG = get_minio_config()

# 创建MinIO客户端实例
minio_client = Minio(
    MINIO_CONFIG['endpoint'],
    access_key=MINIO_CONFIG['access_key'],
    secret_key=MINIO_CONFIG['secret_key'],
    secure=MINIO_CONFIG['secure']
)


def ensure_bucket_exists():
    """确保存储桶存在，如果不存在则创建，并设置为公共读取"""
    try:
        if not minio_client.bucket_exists(MINIO_CONFIG['bucket_name']):
            minio_client.make_bucket(MINIO_CONFIG['bucket_name'])
            print(f"存储桶 '{MINIO_CONFIG['bucket_name']}' 创建成功")

        try:
            import json
            # 设置bucket策略为public-read
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{MINIO_CONFIG['bucket_name']}/*"]
                    }
                ]
            }
            minio_client.set_bucket_policy(
                MINIO_CONFIG['bucket_name'],
                json.dumps(policy)
            )
            print(f"存储桶 '{MINIO_CONFIG['bucket_name']}' 已设置为公共读取")
        except Exception as policy_error:
            print(f"设置存储桶策略时出错（可能需要手动配置）: {policy_error}")
    except S3Error as e:
        print(f"创建存储桶时出错: {e}")
        raise


def upload_file(file_data: bytes, file_name: str, content_type: Optional[str] = None, use_public_url: bool = True) -> str:
    """
    上传文件到MinIO
    
    Args:
        file_data: 文件二进制数据
        file_name: 原始文件名
        content_type: 文件MIME类型
        use_public_url: 是否使用公共URL（True）还是预签名URL（False）
    
    Returns:
        文件的访问URL
    """
    try:
        # 确保存储桶存在并设置了正确的策略
        ensure_bucket_exists()
        
        # 生成唯一文件名（避免文件名冲突）
        file_ext = os.path.splitext(file_name)[1]  # 获取文件扩展名
        file_base_name = os.path.splitext(file_name)[0]  # 获取文件名（不含扩展名）
        
        # 处理文件名：移除或替换特殊字符，限制长度
        # 替换特殊字符为下划线，保留中文字符、字母、数字、连字符和下划线
        safe_file_name = re.sub(r'[^\w\u4e00-\u9fa5-]', '_', file_base_name)
        # 限制文件名长度（避免过长）
        if len(safe_file_name) > 100:
            safe_file_name = safe_file_name[:100]
        
        # 生成唯一标识
        unique_id = str(uuid.uuid4())
        
        # 设置对象路径（可以按日期分类）
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        object_name = f"{date_prefix}-{safe_file_name}-{unique_id}{file_ext}"
        
        # 获取文件大小
        file_size = len(file_data)
        
        # 将 bytes 转换为 BytesIO 对象（MinIO 需要可读对象）
        file_stream = BytesIO(file_data)
        
        # 上传文件
        minio_client.put_object(
            MINIO_CONFIG['bucket_name'],
            object_name,
            data=file_stream,
            length=file_size,
            content_type=content_type or 'application/octet-stream'
        )
        
        # 返回URL
        if use_public_url:
            # 返回公共访问URL（前提是bucket设置为public-read）
            url = f"{MINIO_CONFIG['base_url']}/{MINIO_CONFIG['bucket_name']}/{object_name}"
        else:
            # 使用预签名URL（有效期7天）
            url = minio_client.presigned_get_object(
                MINIO_CONFIG['bucket_name'],
                object_name,
                expires=timedelta(days=7)
            )
        
        return url
        
    except S3Error as e:
        print(f"上传文件到MinIO时出错: {e}")
        raise


def get_file_url(object_name: str, expires_days: int = 7) -> str:
    """
    获取文件的预签名URL
    
    Args:
        object_name: MinIO中的对象名称（路径）
        expires_days: URL有效期（天）
    
    Returns:
        文件的预签名URL
    """
    try:
        url = minio_client.presigned_get_object(
            MINIO_CONFIG['bucket_name'],
            object_name,
            expires=timedelta(days=expires_days)
        )
        return url
    except S3Error as e:
        print(f"获取文件URL时出错: {e}")
        raise


def delete_file(object_name: str) -> bool:
    """
    删除MinIO中的文件
    
    Args:
        object_name: MinIO中的对象名称（路径）
    
    Returns:
        是否删除成功
    """
    try:
        minio_client.remove_object(MINIO_CONFIG['bucket_name'], object_name)
        return True
    except S3Error as e:
        print(f"删除文件时出错: {e}")
        return False
