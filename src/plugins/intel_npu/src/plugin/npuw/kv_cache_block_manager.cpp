// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_block_manager.hpp"

#include "logging.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

KVCacheBlockManager::KVCacheBlockManager(uint32_t block_size,
                                         uint32_t max_blocks,
                                         const ov::Shape& base_shape,
                                         ov::element::Type elem_type,
                                         const std::string& device,
                                         const std::shared_ptr<const ov::IPlugin>& plugin)
    : block_size_(block_size),
      max_blocks_(max_blocks),
      element_type_(elem_type),
      block_shape_(base_shape),
      device_(device),
      plugin_(plugin) {
    OPENVINO_ASSERT(block_size_ > 0 && max_blocks_ > 0,
                    "KVCacheBlockManager: block_size and max_blocks must be > 0, got block_size=",
                    block_size_,
                    " max_blocks=",
                    max_blocks_);
    OPENVINO_ASSERT(device_ == "CPU" || plugin_ != nullptr,
                    "KVCacheBlockManager: plugin must be non-null for non-CPU device '",
                    device_,
                    "'");
    // Check that the sequence dimension (dim 2 for K/non-transposed-V, dim 3 for transposed V)
    // equals block_size.
    OPENVINO_ASSERT(base_shape.size() == 4 && (base_shape[2] == block_size || base_shape[3] == block_size),
                    "KVCacheBlockManager: base_shape ",
                    base_shape,
                    " does not have block_size=",
                    block_size,
                    " in sequence dimension (expected at dim 2 or 3)");

    // Initialize block pool (tensors allocated on-demand, not here)
    blocks_.reserve(max_blocks);
    for (uint32_t i = 0; i < max_blocks; ++i) {
        blocks_.push_back({});
    }

    // Push block IDs to stack in reverse order (so block 0 is on top).
    for (uint32_t i = max_blocks; i > 0; --i) {
        free_block_ids_.push(i - 1);
    }

    LOG_INFO("KVCacheBlockManager initialized: " << "block_size=" << block_size << ", max_blocks=" << max_blocks
                                                 << ", total_capacity=" << (block_size * max_blocks) << " tokens"
                                                 << ", device=" << device);
}

std::optional<uint32_t> KVCacheBlockManager::allocate_block() {
    if (free_block_ids_.empty()) {
        LOG_WARN("KVCacheBlockManager: No free blocks available! " << "All " << max_blocks_ << " blocks are in use.");
        return std::nullopt;
    }

    uint32_t block_id = free_block_ids_.top();
    free_block_ids_.pop();

    auto& block = blocks_[block_id];

    // Allocate actual memory on-demand
    if (!block.tensor) {
        block.tensor = ov::npuw::util::allocMem(element_type_, block_shape_, device_, plugin_);
        LOG_DEBUG("KVCacheBlockManager: Allocated memory for block " << block_id << " (shape=" << block_shape_
                                                                     << ", device=" << device_ << ")");
    }

    // Reset block state; caller owns one reference.
    block.num_tokens = 0;
    block.refcount = 1;

    LOG_VERB("KVCacheBlockManager: Allocated block " << block_id
                                                     << " (free blocks remaining: " << free_block_ids_.size() << ")");

    return block_id;
}

void KVCacheBlockManager::acquire(uint32_t block_id) {
    validate_block_id(block_id);

    auto& block = blocks_[block_id];
    if (block.refcount == 0) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot acquire free block ",
                       block_id,
                       ". Call allocate_block() first.");
    }

    ++block.refcount;
    LOG_VERB("KVCacheBlockManager: Acquired block " << block_id << " (refcount=" << block.refcount << ")");
}

void KVCacheBlockManager::release(uint32_t block_id) {
    validate_block_id(block_id);

    auto& block = blocks_[block_id];
    if (block.refcount == 0) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot release free block ",
                       block_id,
                       ". Double-release or release of never-allocated block.");
    }

    --block.refcount;
    if (block.refcount == 0) {
        // Last reference dropped — return the block to the free pool.
        // Tensor memory is retained for reuse on the next allocate of this id.
        block.num_tokens = 0;
        free_block_ids_.push(block_id);
        LOG_VERB("KVCacheBlockManager: Released block " << block_id
                                                        << " back to free pool (free blocks: " << free_block_ids_.size()
                                                        << ")");
    } else {
        LOG_VERB("KVCacheBlockManager: Released block " << block_id << " (refcount=" << block.refcount << ")");
    }
}

uint32_t KVCacheBlockManager::refcount(uint32_t block_id) const {
    validate_block_id(block_id);
    return blocks_[block_id].refcount;
}

ov::SoPtr<ov::ITensor> KVCacheBlockManager::get_block_tensor(uint32_t block_id) const {
    validate_block_id(block_id);

    auto& block = blocks_[block_id];

    if (!block.tensor) {
        OPENVINO_THROW("KVCacheBlockManager: Block ",
                       block_id,
                       " has no allocated tensor. "
                       "Call allocate_block() first.");
    }

    return block.tensor;
}

void KVCacheBlockManager::update_block_tokens(uint32_t block_id, uint32_t num_tokens) {
    validate_block_id(block_id);

    if (blocks_[block_id].refcount == 0) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot update tokens on free block ",
                       block_id,
                       ". Call allocate_block() first.");
    }

    if (num_tokens > block_size_) {
        OPENVINO_THROW("KVCacheBlockManager: Cannot set ",
                       num_tokens,
                       " tokens in block ",
                       block_id,
                       " (capacity: ",
                       block_size_,
                       ")");
    }

    auto& block = blocks_[block_id];
    block.num_tokens = num_tokens;

    LOG_VERB("KVCacheBlockManager: Updated block " << block_id << " tokens: " << num_tokens << "/" << block_size_);
}

uint32_t KVCacheBlockManager::get_block_tokens(uint32_t block_id) const {
    validate_block_id(block_id);
    return blocks_[block_id].num_tokens;
}

std::vector<uint32_t> KVCacheBlockManager::get_allocated_blocks() const {
    std::vector<uint32_t> allocated;
    allocated.reserve(max_blocks_ - free_block_ids_.size());

    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].refcount > 0) {
            allocated.push_back(i);
        }
    }

    return allocated;
}

void KVCacheBlockManager::clear_all() {
    LOG_DEBUG("KVCacheBlockManager: Clearing all blocks");

    // Count outstanding refs for diagnostics. clear_all() is a force-reset:
    // outstanding refs are likely a caller bug (missing release()), but the
    // contract is to wipe the pool unconditionally.
    uint32_t outstanding = 0;
    for (const auto& block : blocks_) {
        if (block.refcount > 0) {
            ++outstanding;
        }
    }
    if (outstanding > 0) {
        LOG_WARN("KVCacheBlockManager: clear_all() invoked with "
                 << outstanding << " block(s) still holding references. "
                 << "Forcing reset; ensure callers release blocks before clear_all().");
    }

    for (auto& block : blocks_) {
        block.num_tokens = 0;
        block.refcount = 0;
    }

    // Rebuild free stack (push in reverse order so block 0 is on top)
    std::stack<uint32_t> empty;
    free_block_ids_.swap(empty);
    for (uint32_t i = max_blocks_; i > 0; --i) {
        free_block_ids_.push(i - 1);
    }
}

void KVCacheBlockManager::validate_block_id(uint32_t block_id) const {
    if (block_id >= max_blocks_) {
        OPENVINO_THROW("KVCacheBlockManager: Invalid block ID ", block_id, " (valid range: 0-", max_blocks_ - 1, ")");
    }
}

}  // namespace npuw
}  // namespace ov
