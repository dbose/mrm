"""Evidence packet data structure with hash chain semantics

Evidence packets are immutable records of model validation activities.
Each packet contains:
- Model metadata (name, version, artifact hash)
- Test results snapshot
- Compliance paragraph mappings
- Timestamps and human identifiers
- Content hash (SHA-256)
- Reference to prior packet hash (hash chain)
- Optional GPG signature for non-repudiation
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class EvidencePacket:
    """Immutable evidence packet with hash chain semantics
    
    Attributes:
        packet_id: Unique identifier (UUID)
        model_name: Name of the model being validated
        model_version: Version of the model
        model_artifact_hash: SHA-256 hash of model artifact file
        test_results: Dictionary of test results
        compliance_mappings: Paragraph mappings to regulatory standards
        timestamp: ISO 8601 timestamp of packet creation
        created_by: Human identifier (email or username)
        prior_packet_hash: SHA-256 hash of previous packet in chain (None for first)
        content_hash: SHA-256 hash of packet content (computed)
        signature: Optional GPG signature for non-repudiation
        metadata: Additional metadata (profile, trigger context, etc.)
    """
    
    packet_id: str
    model_name: str
    model_version: str
    model_artifact_hash: str
    test_results: Dict[str, Any]
    compliance_mappings: Dict[str, List[str]]
    timestamp: str
    created_by: str
    prior_packet_hash: Optional[str] = None
    content_hash: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute content hash after initialization"""
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()
    
    def _compute_content_hash(self) -> str:
        """Compute SHA-256 hash of packet content
        
        Hash includes all fields except content_hash and signature.
        This ensures the hash is deterministic and verifiable.
        """
        # Create a copy of the packet dict without content_hash and signature
        packet_dict = asdict(self)
        packet_dict.pop('content_hash', None)
        packet_dict.pop('signature', None)
        
        # Serialize to canonical JSON (sorted keys, no whitespace)
        canonical_json = json.dumps(packet_dict, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    def verify_hash(self) -> bool:
        """Verify that content_hash matches recomputed hash
        
        Returns:
            True if hash is valid, False otherwise
        """
        if self.content_hash is None:
            return False
        
        recomputed_hash = self._compute_content_hash()
        return self.content_hash == recomputed_hash
    
    def verify_chain(self, prior_packet: Optional['EvidencePacket']) -> bool:
        """Verify hash chain linkage to prior packet
        
        Args:
            prior_packet: Previous packet in the chain (None if this is first)
        
        Returns:
            True if chain is valid, False otherwise
        """
        if prior_packet is None:
            # First packet in chain should have no prior reference
            return self.prior_packet_hash is None
        
        # Check that our prior_packet_hash matches the prior packet's content_hash
        return self.prior_packet_hash == prior_packet.content_hash
    
    def sign(self, private_key_path: Path, passphrase: Optional[str] = None) -> None:
        """Sign the packet with GPG private key
        
        Args:
            private_key_path: Path to GPG private key file
            passphrase: Optional passphrase for encrypted key
        
        Raises:
            ImportError: If gnupg library not available
            ValueError: If signing fails
        """
        try:
            import gnupg
        except ImportError:
            raise ImportError(
                "GPG signing requires python-gnupg: pip install python-gnupg"
            )
        
        gpg = gnupg.GPG()
        
        # Import private key if needed
        if private_key_path.exists():
            with open(private_key_path, 'r') as f:
                gpg.import_keys(f.read())
        
        # Sign the content hash
        signed = gpg.sign(
            self.content_hash,
            passphrase=passphrase,
            detach=True,
            clearsign=False
        )
        
        if not signed:
            raise ValueError(f"GPG signing failed: {signed.stderr}")
        
        self.signature = str(signed)
    
    def verify_signature(self, public_key_path: Optional[Path] = None) -> bool:
        """Verify GPG signature
        
        Args:
            public_key_path: Optional path to GPG public key file
        
        Returns:
            True if signature is valid, False otherwise
        """
        if self.signature is None:
            return False
        
        try:
            import gnupg
        except ImportError:
            raise ImportError(
                "GPG verification requires python-gnupg: pip install python-gnupg"
            )
        
        gpg = gnupg.GPG()
        
        # Import public key if provided
        if public_key_path and public_key_path.exists():
            with open(public_key_path, 'r') as f:
                gpg.import_keys(f.read())
        
        # Verify signature
        verified = gpg.verify_data(self.content_hash, self.signature)
        return verified.valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize packet to JSON
        
        Args:
            indent: JSON indentation (None for compact)
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidencePacket':
        """Create packet from dictionary
        
        Args:
            data: Dictionary representation
        
        Returns:
            EvidencePacket instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EvidencePacket':
        """Deserialize packet from JSON
        
        Args:
            json_str: JSON string
        
        Returns:
            EvidencePacket instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @staticmethod
    def compute_artifact_hash(artifact_path: Path) -> str:
        """Compute SHA-256 hash of model artifact file
        
        Args:
            artifact_path: Path to model artifact (pickle, h5, etc.)
        
        Returns:
            Hex-encoded SHA-256 hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(artifact_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    @classmethod
    def create(
        cls,
        model_name: str,
        model_version: str,
        model_artifact_path: Path,
        test_results: Dict[str, Any],
        compliance_mappings: Dict[str, List[str]],
        created_by: str,
        prior_packet: Optional['EvidencePacket'] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'EvidencePacket':
        """Factory method to create a new evidence packet
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_artifact_path: Path to model artifact file
            test_results: Dictionary of test results
            compliance_mappings: Paragraph mappings to standards
            created_by: Human identifier (email or username)
            prior_packet: Previous packet in chain (None for first)
            metadata: Additional metadata
        
        Returns:
            New EvidencePacket instance
        """
        import uuid
        
        packet_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        model_artifact_hash = cls.compute_artifact_hash(model_artifact_path)
        prior_packet_hash = prior_packet.content_hash if prior_packet else None
        
        return cls(
            packet_id=packet_id,
            model_name=model_name,
            model_version=model_version,
            model_artifact_hash=model_artifact_hash,
            test_results=test_results,
            compliance_mappings=compliance_mappings,
            timestamp=timestamp,
            created_by=created_by,
            prior_packet_hash=prior_packet_hash,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_for_llm_endpoint(
        cls,
        model_name: str,
        model_version: str,
        provider: str,
        llm_model_name: str,
        test_results: Dict[str, Any],
        compliance_mappings: Dict[str, List[str]],
        created_by: str,
        prior_packet: Optional['EvidencePacket'] = None,
        token_usage: Optional[Dict[str, int]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        prompt_version: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> 'EvidencePacket':
        """Factory method to create evidence packet for LLM endpoint
        
        For LLM endpoints (OpenAI, Anthropic, Bedrock, etc.), there is no
        local artifact file. Instead, we hash the LLM configuration to create
        a pseudo-artifact-hash for versioning purposes.
        
        Args:
            model_name: Name of the mrm model
            model_version: Version of the mrm model
            provider: LLM provider (openai, anthropic, bedrock, etc.)
            llm_model_name: Provider's model identifier (gpt-4, claude-3, etc.)
            test_results: Dictionary of test results
            compliance_mappings: Paragraph mappings to standards
            created_by: Human identifier (email or username)
            prior_packet: Previous packet in chain (None for first)
            token_usage: Dict with prompt_tokens, completion_tokens, total_tokens
            llm_config: LLM parameters (temperature, max_tokens, etc.)
            prompt_version: Version identifier for prompt template
            embedding_model: Embedding model name (for RAG systems)
        
        Returns:
            New EvidencePacket instance with LLM metadata
        """
        import uuid
        
        packet_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Create configuration string for hashing
        config_str = json.dumps({
            'provider': provider,
            'model_name': llm_model_name,
            'config': llm_config or {},
            'prompt_version': prompt_version,
            'embedding_model': embedding_model
        }, sort_keys=True)
        
        # Compute hash of configuration (pseudo-artifact-hash)
        model_artifact_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        
        prior_packet_hash = prior_packet.content_hash if prior_packet else None
        
        # Build LLM-specific metadata
        llm_metadata = {
            'model_type': 'llm_endpoint',
            'provider': provider,
            'llm_model_name': llm_model_name,
            'llm_config': llm_config or {},
            'prompt_version': prompt_version,
            'embedding_model': embedding_model,
            'token_usage': token_usage or {}
        }
        
        return cls(
            packet_id=packet_id,
            model_name=model_name,
            model_version=model_version,
            model_artifact_hash=model_artifact_hash,
            test_results=test_results,
            compliance_mappings=compliance_mappings,
            timestamp=timestamp,
            created_by=created_by,
            prior_packet_hash=prior_packet_hash,
            metadata=llm_metadata
        )
