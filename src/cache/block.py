"""KV-cache block structures for PagedAttention-style memory management.

In standard transformer inference, the KV-cache grows contiguously per
sequence. This wastes memory through fragmentation — short sequences
reserve space for max_seq_len, and completed sequences leave gaps.

PagedAttention (vLLM) solves this by managing KV-cache in fixed-size
blocks, similar to virtual memory pages in an OS:

  Physical blocks (GPU memory) ←→ Logical blocks (per sequence)

Each block holds KV tensors for a fixed number of tokens (block_size).
Sequences map logical block indices to physical blocks via a block table,
allowing non-contiguous memory allocation and efficient sharing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockStatus(Enum):
    """Status of a physical memory block."""
    FREE = "free"
    ALLOCATED = "allocated"
    RESERVED = "reserved"  # pre-allocated but not yet written


@dataclass
class PhysicalBlock:
    """A fixed-size block of GPU/CPU memory holding KV-cache data.

    Each block stores key and value tensors for `block_size` tokens
    across all attention layers and heads.

    Memory layout per block:
        K: [num_layers, num_heads, block_size, head_dim]
        V: [num_layers, num_heads, block_size, head_dim]
    """
    block_id: int
    block_size: int
    status: BlockStatus = BlockStatus.FREE
    ref_count: int = 0  # number of sequences referencing this block
    num_tokens_filled: int = 0  # how many token slots are used

    @property
    def is_full(self) -> bool:
        return self.num_tokens_filled >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.num_tokens_filled

    def allocate(self):
        """Mark block as allocated."""
        self.status = BlockStatus.ALLOCATED
        self.ref_count += 1

    def release(self):
        """Decrement reference count. Free if no references remain."""
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 0:
            self.status = BlockStatus.FREE
            self.num_tokens_filled = 0

    def __repr__(self) -> str:
        return (
            f"PhysicalBlock(id={self.block_id}, "
            f"status={self.status.value}, "
            f"filled={self.num_tokens_filled}/{self.block_size}, "
            f"refs={self.ref_count})"
        )


@dataclass
class BlockTable:
    """Maps a sequence's logical blocks to physical blocks.

    Each sequence maintains its own block table. As the sequence
    generates more tokens, new physical blocks are appended.

    Example for a sequence with 35 tokens and block_size=16:
        logical[0] → physical[5]   (tokens 0-15)
        logical[1] → physical[12]  (tokens 16-31)
        logical[2] → physical[3]   (tokens 32-34, 13 slots free)
    """
    sequence_id: str
    block_size: int
    physical_block_ids: list = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    @property
    def max_tokens(self) -> int:
        """Maximum tokens this block table can hold."""
        return self.num_blocks * self.block_size

    def append_block(self, physical_block_id: int):
        """Map a new physical block to the next logical index."""
        self.physical_block_ids.append(physical_block_id)

    def get_physical_block(self, logical_index: int) -> Optional[int]:
        """Get physical block ID for a logical index."""
        if logical_index < 0 or logical_index >= self.num_blocks:
            return None
        return self.physical_block_ids[logical_index]

    def __repr__(self) -> str:
        mapping = ", ".join(
            f"L{i}→P{pid}" for i, pid in enumerate(self.physical_block_ids)
        )
        return f"BlockTable(seq={self.sequence_id}, [{mapping}])"
