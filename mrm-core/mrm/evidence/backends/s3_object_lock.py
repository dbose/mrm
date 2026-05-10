"""S3 Object Lock evidence backend (PRODUCTION)

AWS S3 with Object Lock in Compliance mode provides regulatory-grade
immutable storage:

✅ SEC 17a-4 compliant (Cohasset Associates assessment)
✅ FINRA Rule 4511 compliant
✅ CFTC Regulation 1.31 compliant
✅ Immutability: Objects cannot be deleted or modified during retention
✅ Audit trail: CloudTrail logs all access
✅ Versioning: S3 maintains version history

Requirements:
- boto3 library: pip install boto3
- S3 bucket with Object Lock enabled (must be set at bucket creation)
- IAM permissions: s3:PutObject, s3:PutObjectRetention, s3:GetObject

Storage layout:
    s3://{bucket}/evidence/{model_name}/{packet_id}.json
    
Each packet is stored as a separate S3 object with Object Lock retention
applied. Compliance mode ensures objects cannot be deleted even by root.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from mrm.evidence.base import EvidenceBackend
from mrm.evidence.packet import EvidencePacket

logger = logging.getLogger(__name__)


class S3ObjectLockBackend(EvidenceBackend):
    """S3 Object Lock evidence storage (PRODUCTION)
    
    Attributes:
        bucket: S3 bucket name (must have Object Lock enabled)
        prefix: Optional S3 key prefix (default: "evidence/")
        retention_mode: COMPLIANCE or GOVERNANCE (default: COMPLIANCE)
        default_retention_days: Default retention period in days
        region: AWS region
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "evidence/",
        retention_mode: str = "COMPLIANCE",
        default_retention_days: int = 2555,  # ~7 years
        region: Optional[str] = None,
        **boto3_kwargs
    ):
        """Initialize S3 Object Lock backend
        
        Args:
            bucket: S3 bucket name (must have Object Lock enabled)
            prefix: S3 key prefix for evidence objects
            retention_mode: COMPLIANCE (cannot delete) or GOVERNANCE (can with permission)
            default_retention_days: Default retention period in days
            region: AWS region (uses default if None)
            **boto3_kwargs: Additional arguments passed to boto3.client()
        
        Raises:
            ImportError: If boto3 not installed
            ValueError: If invalid retention_mode
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "S3 backend requires boto3: pip install boto3"
            )
        
        if retention_mode not in ('COMPLIANCE', 'GOVERNANCE'):
            raise ValueError(
                f"retention_mode must be COMPLIANCE or GOVERNANCE, got: {retention_mode}"
            )
        
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        self.retention_mode = retention_mode
        self.default_retention_days = default_retention_days
        self.region = region
        
        # Initialize S3 client
        client_kwargs = {'region_name': region} if region else {}
        client_kwargs.update(boto3_kwargs)
        self.s3 = boto3.client('s3', **client_kwargs)
        
        # Verify bucket exists and has Object Lock enabled
        self._verify_bucket_config()
        
        logger.info(
            f"Initialized S3ObjectLockBackend: s3://{bucket}/{prefix} "
            f"(mode={retention_mode}, retention={default_retention_days}d)"
        )
    
    def _verify_bucket_config(self) -> None:
        """Verify S3 bucket exists and has Object Lock enabled
        
        Raises:
            ValueError: If bucket doesn't exist or Object Lock not enabled
        """
        try:
            # Check if Object Lock is enabled
            response = self.s3.get_object_lock_configuration(Bucket=self.bucket)
            
            if response['ObjectLockConfiguration']['ObjectLockEnabled'] != 'Enabled':
                raise ValueError(
                    f"Bucket {self.bucket} does not have Object Lock enabled"
                )
            
            logger.info(f"Verified bucket {self.bucket} has Object Lock enabled")
        
        except self.s3.exceptions.NoSuchBucket:
            raise ValueError(f"S3 bucket does not exist: {self.bucket}")
        
        except self.s3.exceptions.ObjectLockConfigurationNotFoundError:
            raise ValueError(
                f"Bucket {self.bucket} does not have Object Lock enabled. "
                "Object Lock must be enabled at bucket creation time."
            )
    
    def _get_s3_key(self, model_name: str, packet_id: str) -> str:
        """Generate S3 key for evidence packet"""
        return f"{self.prefix}{model_name}/{packet_id}.json"
    
    def _parse_s3_uri(self, uri: str) -> tuple:
        """Parse s3:// URI into bucket and key
        
        Returns:
            (bucket, key) tuple
        """
        if not uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {uri}")
        
        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {uri}")
        
        return parts[0], parts[1]
    
    def freeze(
        self,
        packet: EvidencePacket,
        retention_days: Optional[int] = None
    ) -> str:
        """Store evidence packet with Object Lock retention
        
        Args:
            packet: Evidence packet to store
            retention_days: Retention period in days (uses default if None)
        
        Returns:
            S3 URI: s3://{bucket}/{prefix}{model_name}/{packet_id}.json
        """
        # Verify packet hash before storing
        if not packet.verify_hash():
            raise ValueError(f"Packet {packet.packet_id} failed hash verification")
        
        # Calculate retention date
        retention_days = retention_days or self.default_retention_days
        retain_until = datetime.utcnow() + timedelta(days=retention_days)
        
        # Generate S3 key
        s3_key = self._get_s3_key(packet.model_name, packet.packet_id)
        
        # Upload to S3 with Object Lock retention
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=packet.to_json(indent=2).encode('utf-8'),
                ContentType='application/json',
                ObjectLockMode=self.retention_mode,
                ObjectLockRetainUntilDate=retain_until,
                Metadata={
                    'model-name': packet.model_name,
                    'model-version': packet.model_version,
                    'packet-id': packet.packet_id,
                    'created-by': packet.created_by,
                    'timestamp': packet.timestamp,
                    'content-hash': packet.content_hash,
                    'prior-packet-hash': packet.prior_packet_hash or ''
                }
            )
        except Exception as e:
            logger.error(f"Failed to freeze packet {packet.packet_id}: {e}")
            raise IOError(f"S3 upload failed: {e}")
        
        uri = f"s3://{self.bucket}/{s3_key}"
        
        logger.info(
            f"Froze evidence packet: {uri} "
            f"(retention={retention_days}d, mode={self.retention_mode})"
        )
        
        return uri
    
    def retrieve(self, packet_uri: str) -> EvidencePacket:
        """Retrieve evidence packet from S3
        
        Args:
            packet_uri: s3://{bucket}/{key}
        
        Returns:
            Evidence packet
        """
        bucket, key = self._parse_s3_uri(packet_uri)
        
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            packet_json = response['Body'].read().decode('utf-8')
            packet = EvidencePacket.from_json(packet_json)
            
            # Verify hash on retrieval
            if not packet.verify_hash():
                raise ValueError(
                    f"Packet {packet.packet_id} failed hash verification - possible corruption"
                )
            
            return packet
        
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Packet not found: {packet_uri}")
        except Exception as e:
            logger.error(f"Failed to retrieve packet {packet_uri}: {e}")
            raise
    
    def list_packets(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List evidence packets from S3
        
        Args:
            model_name: Filter by model name
            start_date: Filter by start date (ISO 8601)
            end_date: Filter by end date (ISO 8601)
        
        Returns:
            List of packet metadata dicts
        """
        results = []
        
        # Determine S3 prefix for listing
        if model_name:
            list_prefix = f"{self.prefix}{model_name}/"
        else:
            list_prefix = self.prefix
        
        # List objects
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=list_prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                
                try:
                    # Get object metadata
                    response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
                    metadata = response.get('Metadata', {})
                    
                    timestamp = metadata.get('timestamp', '')
                    
                    # Apply date filters
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    
                    packet_meta = {
                        'uri': f"s3://{self.bucket}/{s3_key}",
                        'packet_id': metadata.get('packet-id', ''),
                        'model_name': metadata.get('model-name', ''),
                        'model_version': metadata.get('model-version', ''),
                        'timestamp': timestamp,
                        'created_by': metadata.get('created-by', ''),
                        'content_hash': metadata.get('content-hash', ''),
                        'size_bytes': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    }
                    
                    results.append(packet_meta)
                
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {s3_key}: {e}")
                    continue
        
        # Sort by timestamp descending
        results.sort(key=lambda p: p.get('timestamp', ''), reverse=True)
        
        return results
    
    def verify(self, packet_uri: str, verify_chain: bool = True) -> Dict[str, Any]:
        """Verify packet integrity and optionally hash chain
        
        Args:
            packet_uri: S3 URI of packet
            verify_chain: If True, verify entire chain from this packet back
        
        Returns:
            Verification results dict
        """
        try:
            packet = self.retrieve(packet_uri)
        except (FileNotFoundError, ValueError) as e:
            return {
                'valid': False,
                'reason': str(e),
                'packet_uri': packet_uri
            }
        
        # Verify packet hash
        if not packet.verify_hash():
            return {
                'valid': False,
                'reason': 'Packet hash verification failed',
                'packet_id': packet.packet_id
            }
        
        # Check S3 Object Lock status
        try:
            bucket, key = self._parse_s3_uri(packet_uri)
            response = self.s3.get_object_retention(Bucket=bucket, Key=key)
            
            retention_info = {
                'retention_mode': response.get('Retention', {}).get('Mode'),
                'retain_until': response.get('Retention', {}).get('RetainUntilDate', '').isoformat() if response.get('Retention', {}).get('RetainUntilDate') else None
            }
        except Exception as e:
            retention_info = {'error': str(e)}
        
        # If verify_chain, trace back to first packet
        if verify_chain:
            chain_result = self.verify_full_chain(packet.model_name)
            chain_result['retention_info'] = retention_info
            return chain_result
        
        return {
            'valid': True,
            'reason': 'Packet hash valid',
            'packet_id': packet.packet_id,
            'retention_info': retention_info
        }
