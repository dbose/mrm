"""Evidence storage backends"""

from mrm.evidence.backends.local import LocalFilesystemBackend

__all__ = ['LocalFilesystemBackend']

# S3 backend is optional - only import if boto3 available
try:
    from mrm.evidence.backends.s3_object_lock import S3ObjectLockBackend
    __all__.append('S3ObjectLockBackend')
except ImportError:
    pass
