"""Local filesystem evidence backend (DEV/TESTING ONLY)

⚠️  WARNING: NOT FOR REGULATORY USE ⚠️

This backend stores evidence packets as JSONL (JSON Lines) files on the
local filesystem. It is suitable ONLY for:
- Development and testing
- Proof-of-concept demonstrations 
- CI/CD validation pipelines

DO NOT USE for regulated production environments. Reasons:
1. No immutability guarantees (files can be deleted/modified)
2. No SEC 17a-4 / FINRA compliance
3. No audit log of access/modifications
4. No protection against malicious tampering

For production use, deploy S3 Object Lock backend.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from mrm.evidence.base import EvidenceBackend
from mrm.evidence.packet import EvidencePacket

logger = logging.getLogger(__name__)


class LocalFilesystemBackend(EvidenceBackend):
    """Local filesystem evidence storage (DEV ONLY)
    
    Storage layout:
        evidence_dir/
            {model_name}/
                packets.jsonl      # Append-only JSONL file
                index.json         # Packet metadata index
    
    Each line in packets.jsonl is a complete evidence packet JSON.
    The index.json maintains a lookup table for fast queries.
    
    Attributes:
        evidence_dir: Root directory for evidence storage
        warn_on_use: If True, emit warning on every operation
    """
    
    def __init__(
        self,
        evidence_dir: Path,
        warn_on_use: bool = True
    ):
        """Initialize local filesystem backend
        
        Args:
            evidence_dir: Root directory for evidence storage
            warn_on_use: If True, emit dev-only warnings (recommended)
        """
        self.evidence_dir = Path(evidence_dir)
        self.warn_on_use = warn_on_use
        
        # Create evidence directory if it doesn't exist
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        if self.warn_on_use:
            logger.warning(
                "⚠️  Using LocalFilesystemBackend - NOT FOR REGULATORY USE ⚠️\n"
                "This backend provides NO immutability guarantees and is NOT "
                "compliant with SEC 17a-4, FINRA, or other regulatory requirements.\n"
                "Use S3 Object Lock backend for production environments."
            )
    
    def _get_model_dir(self, model_name: str) -> Path:
        """Get directory for model's evidence packets"""
        model_dir = self.evidence_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _get_packets_file(self, model_name: str) -> Path:
        """Get JSONL file path for model's packets"""
        return self._get_model_dir(model_name) / "packets.jsonl"
    
    def _get_index_file(self, model_name: str) -> Path:
        """Get index file path for model"""
        return self._get_model_dir(model_name) / "index.json"
    
    def _load_index(self, model_name: str) -> Dict[str, Any]:
        """Load packet index from disk"""
        index_file = self._get_index_file(model_name)
        
        if not index_file.exists():
            return {'packets': []}
        
        with open(index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self, model_name: str, index: Dict[str, Any]) -> None:
        """Save packet index to disk"""
        index_file = self._get_index_file(model_name)
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def freeze(
        self,
        packet: EvidencePacket,
        retention_days: Optional[int] = None
    ) -> str:
        """Store evidence packet in JSONL file
        
        Args:
            packet: Evidence packet to store
            retention_days: Ignored for local backend
        
        Returns:
            URI: file://{evidence_dir}/{model_name}/packets.jsonl#{packet_id}
        """
        # Verify packet hash before storing
        if not packet.verify_hash():
            raise ValueError(f"Packet {packet.packet_id} failed hash verification")
        
        # Append packet to JSONL file
        packets_file = self._get_packets_file(packet.model_name)
        
        with open(packets_file, 'a') as f:
            f.write(packet.to_json(indent=None) + '\n')
        
        # Update index
        index = self._load_index(packet.model_name)
        index['packets'].append({
            'packet_id': packet.packet_id,
            'model_name': packet.model_name,
            'model_version': packet.model_version,
            'timestamp': packet.timestamp,
            'created_by': packet.created_by,
            'line_number': len(index['packets'])  # 0-indexed
        })
        self._save_index(packet.model_name, index)
        
        # Generate URI
        uri = f"file://{packets_file.absolute()}#{packet.packet_id}"
        
        logger.info(f"Froze evidence packet: {uri}")
        
        if retention_days:
            logger.warning(
                f"retention_days={retention_days} ignored by LocalFilesystemBackend"
            )
        
        return uri
    
    def retrieve(self, packet_uri: str) -> EvidencePacket:
        """Retrieve packet from JSONL file by URI
        
        Args:
            packet_uri: file://{path}/packets.jsonl#{packet_id}
        
        Returns:
            Evidence packet
        """
        # Parse URI
        if not packet_uri.startswith('file://'):
            raise ValueError(f"Invalid URI format: {packet_uri}")
        
        if '#' not in packet_uri:
            raise ValueError(f"URI missing packet_id fragment: {packet_uri}")
        
        path_part, packet_id = packet_uri[7:].split('#', 1)
        packets_file = Path(path_part)
        
        if not packets_file.exists():
            raise FileNotFoundError(f"Packets file not found: {packets_file}")
        
        # Read JSONL and find packet
        with open(packets_file, 'r') as f:
            for line in f:
                packet_dict = json.loads(line)
                if packet_dict.get('packet_id') == packet_id:
                    packet = EvidencePacket.from_dict(packet_dict)
                    
                    # Verify hash on retrieval
                    if not packet.verify_hash():
                        raise ValueError(
                            f"Packet {packet_id} failed hash verification - possible tampering"
                        )
                    
                    return packet
        
        raise FileNotFoundError(f"Packet {packet_id} not found in {packets_file}")
    
    def list_packets(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List evidence packets with optional filters
        
        Args:
            model_name: Filter by model name
            start_date: Filter by start date (ISO 8601)
            end_date: Filter by end date (ISO 8601)
        
        Returns:
            List of packet metadata dicts
        """
        results = []
        
        # Determine which models to search
        if model_name:
            model_dirs = [self._get_model_dir(model_name)]
        else:
            model_dirs = [d for d in self.evidence_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            index_file = model_dir / "index.json"
            if not index_file.exists():
                continue
            
            # Load index with error handling
            try:
                with open(index_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file, skip
                        continue
                    index = json.loads(content)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupt index file {index_file}: {e}")
                continue
            
            for packet_meta in index.get('packets', []):
                timestamp = packet_meta.get('timestamp', '')
                
                # Apply date filters
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                
                # Generate URI
                packets_file = model_dir / "packets.jsonl"
                packet_meta['uri'] = f"file://{packets_file.absolute()}#{packet_meta['packet_id']}"
                
                results.append(packet_meta)
        
        # Sort by timestamp descending
        results.sort(key=lambda p: p.get('timestamp', ''), reverse=True)
        
        return results
    
    def verify(self, packet_uri: str, verify_chain: bool = True) -> Dict[str, Any]:
        """Verify packet integrity and optionally hash chain
        
        Args:
            packet_uri: URI of packet to verify
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
        
        # If verify_chain, trace back to first packet
        if verify_chain:
            return self.verify_full_chain(packet.model_name)
        
        return {
            'valid': True,
            'reason': 'Packet hash valid',
            'packet_id': packet.packet_id
        }
