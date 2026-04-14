"""KV-cache memory manager.

Manages a pool of physical blocks and handles allocation/deallocation
for sequences. This is the core memory management layer — analogous
to a page allocator in an operating system.

Key responsibilities:
  1. Maintain a free list of physical blocks
  2. Allocate blocks to sequences on demand
  3. Track per-sequence memory usage via block tables
  4. Evict sequences when memory is full (LRU policy)
  5. Report memory utilization metrics
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.cache.block import BlockStatus, BlockTable, PhysicalBlock


class CacheManager:
    """Manages KV-cache block allocation across all active sequences.

    Args:
        num_blocks: Total number of physical blocks in the pool.
        block_size: Number of tokens each block can hold.
    """

    def __init__(self, num_blocks: int = 256, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Initialize physical block pool
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        ]

        # Free block indices (LIFO for cache locality)
        self._free_block_ids: List[int] = list(range(num_blocks))

        # Per-sequence block tables
        self._block_tables: Dict[str, BlockTable] = {}

        # LRU tracking: sequence_id → last access time
        self._access_order: OrderedDict[str, float] = OrderedDict()

        # Stats
        self._total_allocated = 0
        self._total_evicted = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        """Memory utilization as a fraction (0.0 to 1.0)."""
        if self.num_blocks == 0:
            return 0.0
        return self.num_allocated_blocks / self.num_blocks

    @property
    def num_active_sequences(self) -> int:
        return len(self._block_tables)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we have enough free blocks for the given token count."""
        blocks_needed = self._tokens_to_blocks(num_tokens)
        return blocks_needed <= self.num_free_blocks

    def allocate_sequence(self, sequence_id: str, num_tokens: int) -> BlockTable:
        """Allocate blocks for a new sequence.

        Args:
            sequence_id: Unique identifier for the sequence.
            num_tokens: Number of tokens to allocate space for.

        Returns:
            BlockTable mapping logical to physical blocks.

        Raises:
            MemoryError: If not enough free blocks available.
        """
        blocks_needed = self._tokens_to_blocks(num_tokens)

        if blocks_needed > self.num_free_blocks:
            raise MemoryError(
                f"Cannot allocate {blocks_needed} blocks for sequence "
                f"{sequence_id}. Only {self.num_free_blocks} free blocks."
            )

        # Create block table
        block_table = BlockTable(
            sequence_id=sequence_id,
            block_size=self.block_size,
        )

        # Allocate physical blocks
        for i in range(blocks_needed):
            block_id = self._free_block_ids.pop()
            block = self.blocks[block_id]
            block.allocate()

            # Set tokens filled
            remaining = num_tokens - (i * self.block_size)
            block.num_tokens_filled = min(remaining, self.block_size)

            block_table.append_block(block_id)

        self._block_tables[sequence_id] = block_table
        self._touch(sequence_id)
        self._total_allocated += blocks_needed

        logger.debug(
            f"Allocated {blocks_needed} blocks for seq {sequence_id} "
            f"({self.num_free_blocks} free remaining)"
        )

        return block_table

    def extend_sequence(self, sequence_id: str, num_new_tokens: int) -> int:
        """Allocate additional blocks for a growing sequence.

        When a sequence generates more tokens and fills its last block,
        we need to allocate new blocks.

        Args:
            sequence_id: Sequence to extend.
            num_new_tokens: Additional tokens to accommodate.

        Returns:
            Number of new blocks allocated.
        """
        if sequence_id not in self._block_tables:
            raise KeyError(f"Sequence {sequence_id} not found")

        block_table = self._block_tables[sequence_id]
        self._touch(sequence_id)

        # Check if last block has room
        new_blocks_needed = 0
        tokens_remaining = num_new_tokens

        if block_table.num_blocks > 0:
            last_block_id = block_table.physical_block_ids[-1]
            last_block = self.blocks[last_block_id]
            can_fit = last_block.free_slots
            if can_fit > 0:
                fit = min(can_fit, tokens_remaining)
                last_block.num_tokens_filled += fit
                tokens_remaining -= fit

        if tokens_remaining > 0:
            new_blocks_needed = self._tokens_to_blocks(tokens_remaining)

        if new_blocks_needed > self.num_free_blocks:
            raise MemoryError(
                f"Cannot extend seq {sequence_id} by {new_blocks_needed} blocks"
            )

        for i in range(new_blocks_needed):
            block_id = self._free_block_ids.pop()
            block = self.blocks[block_id]
            block.allocate()

            remaining = tokens_remaining - (i * self.block_size)
            block.num_tokens_filled = min(remaining, self.block_size)

            block_table.append_block(block_id)

        self._total_allocated += new_blocks_needed
        return new_blocks_needed

    def free_sequence(self, sequence_id: str):
        """Release all blocks allocated to a sequence.

        Args:
            sequence_id: Sequence to free.
        """
        if sequence_id not in self._block_tables:
            return

        block_table = self._block_tables[sequence_id]
        for block_id in block_table.physical_block_ids:
            block = self.blocks[block_id]
            block.release()
            if block.status == BlockStatus.FREE:
                self._free_block_ids.append(block_id)

        del self._block_tables[sequence_id]
        self._access_order.pop(sequence_id, None)

        logger.debug(
            f"Freed {block_table.num_blocks} blocks from seq {sequence_id}"
        )

    def evict_lru(self) -> Optional[str]:
        """Evict the least recently used sequence.

        Returns:
            The sequence_id that was evicted, or None if no sequences.
        """
        if not self._access_order:
            return None

        # Pop the oldest entry
        victim_id, _ = self._access_order.popitem(last=False)
        blocks_freed = self._block_tables[victim_id].num_blocks
        self.free_sequence(victim_id)
        self._total_evicted += 1

        logger.info(
            f"Evicted seq {victim_id} ({blocks_freed} blocks freed)"
        )
        return victim_id

    def get_block_table(self, sequence_id: str) -> Optional[BlockTable]:
        """Get the block table for a sequence."""
        self._touch(sequence_id)
        return self._block_tables.get(sequence_id)

    def get_stats(self) -> dict:
        """Return cache manager statistics."""
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "allocated_blocks": self.num_allocated_blocks,
            "utilization": round(self.utilization, 3),
            "active_sequences": self.num_active_sequences,
            "total_allocated_lifetime": self._total_allocated,
            "total_evictions": self._total_evicted,
            "block_size": self.block_size,
        }

    def _tokens_to_blocks(self, num_tokens: int) -> int:
        """Calculate how many blocks are needed for a token count."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def _touch(self, sequence_id: str):
        """Update LRU access time for a sequence."""
        if sequence_id in self._access_order:
            self._access_order.move_to_end(sequence_id)
        self._access_order[sequence_id] = time.monotonic()
