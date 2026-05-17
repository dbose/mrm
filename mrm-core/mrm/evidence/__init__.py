"""Evidence vault for immutable model validation evidence packets.

The vault has three layers:

  * ``packet``  -- single-event ``EvidencePacket`` with hash chain
                   linkage (P5).
  * ``chain``   -- HMAC-chained event log (the *fast path* in P9).
  * ``merkle``  -- daily Merkle aggregation of a chain (P9 bridge to
                   the lockdown path).
  * ``sign``    -- pluggable signer (LocalSigner / GpgSigner /
                   AgeSigner / KmsSigner OSS; CloudHsmSigner paid).
"""

from mrm.evidence.packet import EvidencePacket
from mrm.evidence.base import EvidenceBackend
from mrm.evidence.chain import (
    ChainedEvent,
    ChainReader,
    ChainWriter,
    derive_session_key,
    load_or_create_chain_secret,
)
from mrm.evidence.merkle import (
    DailyMerkleRoot,
    aggregate_epoch,
    leaf_hash,
    merkle_root,
    node_hash,
    read_root,
    reproduce_root_from_chain,
    write_root,
)
from mrm.evidence.sign import (
    AgeSigner,
    CloudHsmSigner,
    GpgSigner,
    KmsSigner,
    LocalSigner,
    Signer,
    build_signer,
    get_signer_cls,
    list_signers,
    register_signer,
)

__all__ = [
    "EvidencePacket",
    "EvidenceBackend",
    "ChainedEvent",
    "ChainReader",
    "ChainWriter",
    "derive_session_key",
    "load_or_create_chain_secret",
    "DailyMerkleRoot",
    "aggregate_epoch",
    "leaf_hash",
    "merkle_root",
    "node_hash",
    "read_root",
    "reproduce_root_from_chain",
    "write_root",
    "Signer",
    "LocalSigner",
    "GpgSigner",
    "AgeSigner",
    "KmsSigner",
    "CloudHsmSigner",
    "build_signer",
    "get_signer_cls",
    "list_signers",
    "register_signer",
]
