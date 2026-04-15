"""Tests for KV-cache block management."""

import pytest

from src.cache.block import BlockStatus, BlockTable, PhysicalBlock
from src.cache.manager import CacheManager


class TestPhysicalBlock:
    def test_initial_state(self):
        block = PhysicalBlock(block_id=0, block_size=16)
        assert block.status == BlockStatus.FREE
        assert block.ref_count == 0
        assert block.free_slots == 16
        assert not block.is_full

    def test_allocate(self):
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        assert block.status == BlockStatus.ALLOCATED
        assert block.ref_count == 1

    def test_release(self):
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        block.release()
        assert block.status == BlockStatus.FREE
        assert block.ref_count == 0
        assert block.num_tokens_filled == 0

    def test_is_full(self):
        block = PhysicalBlock(block_id=0, block_size=4)
        block.num_tokens_filled = 4
        assert block.is_full
        assert block.free_slots == 0


class TestBlockTable:
    def test_append_and_lookup(self):
        bt = BlockTable(sequence_id="seq1", block_size=16)
        bt.append_block(5)
        bt.append_block(12)
        assert bt.num_blocks == 2
        assert bt.get_physical_block(0) == 5
        assert bt.get_physical_block(1) == 12
        assert bt.max_tokens == 32

    def test_out_of_range(self):
        bt = BlockTable(sequence_id="seq1", block_size=16)
        assert bt.get_physical_block(0) is None
        assert bt.get_physical_block(-1) is None


class TestCacheManager:
    def test_initial_state(self):
        mgr = CacheManager(num_blocks=64, block_size=16)
        assert mgr.num_free_blocks == 64
        assert mgr.num_allocated_blocks == 0
        assert mgr.utilization == 0.0

    def test_allocate_sequence(self):
        mgr = CacheManager(num_blocks=64, block_size=16)
        bt = mgr.allocate_sequence("seq1", num_tokens=35)
        # 35 tokens / 16 block_size = 3 blocks
        assert bt.num_blocks == 3
        assert mgr.num_free_blocks == 61
        assert mgr.num_active_sequences == 1

    def test_free_sequence(self):
        mgr = CacheManager(num_blocks=64, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=16)
        mgr.free_sequence("seq1")
        assert mgr.num_free_blocks == 64
        assert mgr.num_active_sequences == 0

    def test_extend_sequence(self):
        mgr = CacheManager(num_blocks=64, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=10)
        # Block has 6 free slots, extending by 6 shouldn't need new block
        new_blocks = mgr.extend_sequence("seq1", num_new_tokens=6)
        assert new_blocks == 0
        assert mgr.num_free_blocks == 63

        # Now extending by 1 more should need a new block
        new_blocks = mgr.extend_sequence("seq1", num_new_tokens=1)
        assert new_blocks == 1
        assert mgr.num_free_blocks == 62

    def test_evict_lru(self):
        mgr = CacheManager(num_blocks=8, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=16)
        mgr.allocate_sequence("seq2", num_tokens=16)
        mgr.allocate_sequence("seq3", num_tokens=16)

        # seq1 is least recently used
        evicted = mgr.evict_lru()
        assert evicted == "seq1"
        assert mgr.num_active_sequences == 2
        assert mgr.num_free_blocks == 6

    def test_evict_lru_respects_access(self):
        mgr = CacheManager(num_blocks=8, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=16)
        mgr.allocate_sequence("seq2", num_tokens=16)

        # Touch seq1 to make it more recent
        mgr.get_block_table("seq1")

        evicted = mgr.evict_lru()
        assert evicted == "seq2"  # seq2 is now least recent

    def test_out_of_memory(self):
        mgr = CacheManager(num_blocks=2, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=32)  # uses all blocks
        with pytest.raises(MemoryError):
            mgr.allocate_sequence("seq2", num_tokens=1)

    def test_can_allocate(self):
        mgr = CacheManager(num_blocks=4, block_size=16)
        assert mgr.can_allocate(64)  # exactly 4 blocks
        assert not mgr.can_allocate(65)  # needs 5 blocks

    def test_stats(self):
        mgr = CacheManager(num_blocks=32, block_size=16)
        mgr.allocate_sequence("seq1", num_tokens=32)
        stats = mgr.get_stats()
        assert stats["total_blocks"] == 32
        assert stats["allocated_blocks"] == 2
        assert stats["active_sequences"] == 1
        assert stats["block_size"] == 16
