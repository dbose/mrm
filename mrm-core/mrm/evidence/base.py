"""Abstract base class for evidence storage backends

All evidence backends must implement:
- freeze: Store an immutable evidence packet
- retrieve: Fetch a packet by URI
- list_packets: List all packets for a model
- verify: Verify packet integrity and hash chain

Backends handle immutability constraints at the storage layer:
- Local: JSONL append-only file with dev warnings
- S3 Object Lock: Compliance-mode retention, Cohasset-assessed for SEC 17a-4
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from mrm.evidence.packet import EvidencePacket


class EvidenceBackend(ABC):
    """Abstract base class for evidence storage backends
    
    Implementations must guarantee:
    1. Immutability: Once written, packets cannot be modified or deleted
    2. Append-only: New packets can only be added to the chain
    3. Verifiability: All packets can be verified for integrity
    4. Auditability: All operations are logged/traceable
    """
    
    @abstractmethod
    def freeze(
        self,
        packet: EvidencePacket,
        retention_days: Optional[int] = None
    ) -> str:
        """Store an immutable evidence packet
        
        Args:
            packet: Evidence packet to store
            retention_days: Optional retention period in days
        
        Returns:
            URI to the stored packet
        
        Raises:
            ValueError: If packet validation fails
            IOError: If storage operation fails
        """
        pass
    
    @abstractmethod
    def retrieve(self, packet_uri: str) -> EvidencePacket:
        """Retrieve an evidence packet by URI
        
        Args:
            packet_uri: URI returned by freeze()
        
        Returns:
            Evidence packet
        
        Raises:
            FileNotFoundError: If packet does not exist
            ValueError: If packet is corrupted
        """
        pass
    
    @abstractmethod
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
            List of packet metadata dicts (uri, model_name, timestamp, etc.)
        """
        pass
    
    @abstractmethod
    def verify(self, packet_uri: str, verify_chain: bool = True) -> Dict[str, Any]:
        """Verify packet integrity and optionally the hash chain
        
        Args:
            packet_uri: URI of packet to verify
            verify_chain: If True, verify entire chain from this packet back
        
        Returns:
            Verification results dict with status and details
        """
        pass
    
    def get_latest_packet(self, model_name: str) -> Optional[EvidencePacket]:
        """Get the most recent packet for a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Latest packet or None if no packets exist
        """
        packets = self.list_packets(model_name=model_name)
        if not packets:
            return None
        
        # Sort by timestamp descending
        packets_sorted = sorted(
            packets,
            key=lambda p: p.get('timestamp', ''),
            reverse=True
        )
        
        latest_uri = packets_sorted[0]['uri']
        return self.retrieve(latest_uri)
    
    def verify_full_chain(self, model_name: str) -> Dict[str, Any]:
        """Verify entire evidence chain for a model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Verification results with chain validity status
        """
        packets_meta = self.list_packets(model_name=model_name)
        if not packets_meta:
            return {
                'valid': True,
                'reason': 'No packets found',
                'packet_count': 0
            }
        
        # Sort by timestamp ascending (oldest first)
        packets_meta = sorted(packets_meta, key=lambda p: p.get('timestamp', ''))
        
        # Retrieve all packets
        packets = [self.retrieve(p['uri']) for p in packets_meta]
        
        # Verify each packet's hash
        for i, packet in enumerate(packets):
            if not packet.verify_hash():
                return {
                    'valid': False,
                    'reason': f'Packet {i} failed hash verification',
                    'packet_id': packet.packet_id,
                    'packet_count': len(packets)
                }
        
        # Verify chain linkage
        for i in range(1, len(packets)):
            prior_packet = packets[i - 1]
            current_packet = packets[i]
            
            if not current_packet.verify_chain(prior_packet):
                return {
                    'valid': False,
                    'reason': f'Chain broken between packet {i-1} and {i}',
                    'prior_packet_id': prior_packet.packet_id,
                    'current_packet_id': current_packet.packet_id,
                    'packet_count': len(packets)
                }
        
        # Verify first packet has no prior reference
        if packets[0].prior_packet_hash is not None:
            return {
                'valid': False,
                'reason': 'First packet has invalid prior reference',
                'packet_id': packets[0].packet_id,
                'packet_count': len(packets)
            }
        
        return {
            'valid': True,
            'reason': 'All packets verified',
            'packet_count': len(packets),
            'first_packet': packets[0].packet_id,
            'latest_packet': packets[-1].packet_id
        }
