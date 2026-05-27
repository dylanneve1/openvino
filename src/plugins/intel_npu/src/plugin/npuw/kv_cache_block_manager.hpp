// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

/**
 * @brief Block-based KV Cache Manager
 *
 * Manages KV cache memory using fixed-size blocks instead of a single continuous buffer.
 * This approach significantly reduces memory waste when actual prompt length is much smaller
 * than the configured max_prompt_size.
 *
 * Key benefits:
 * - Memory efficiency: Only allocate blocks as needed
 * - Flexible allocation: Adapt to variable-length prompts without pre-allocating max buffer
 *
 * Each allocated block carries a reference count (refcount):
 *   - allocate_block() returns a block with refcount = 1 (caller owns one ref).
 *   - acquire(id) increments the refcount (additional owner, e.g. a prefix
 *     cache entry holding the block alongside a live infer request).
 *   - release(id) decrements; when refcount reaches 0 the block returns to
 *     the free pool. Tensor memory is retained for reuse.
 *
 * Example usage:
 *   KVCacheBlockManager manager(512, 16, shape, type, "NPU", plugin);
 *   auto block_id = manager.allocate_block();        // refcount = 1
 *   auto tensor   = manager.get_block_tensor(block_id.value());
 *   manager.update_block_tokens(block_id.value(), 256);
 *   manager.acquire(block_id.value());               // refcount = 2
 *   manager.release(block_id.value());               // refcount = 1
 *   manager.release(block_id.value());               // refcount = 0, freed
 */
class KVCacheBlockManager {
public:
    /**
     * @brief Construct a new KV Cache Block Manager
     *
     * @param block_size Number of tokens per block
     * @param max_blocks Maximum number of blocks in the pool
     * @param base_shape Base shape for block tensors [batch, num_heads, seq_len, head_dim]
     * @param elem_type Element type (e.g., fp16, fp32)
     * @param device Target device for memory allocation ("NPU", "CPU")
     * @param plugin Plugin instance for memory allocation
     */
    KVCacheBlockManager(uint32_t block_size,
                        uint32_t max_blocks,
                        const ov::Shape& base_shape,
                        ov::element::Type elem_type,
                        const std::string& device,
                        const std::shared_ptr<const ov::IPlugin>& plugin);

    ~KVCacheBlockManager() = default;

    // Disable copy
    KVCacheBlockManager(const KVCacheBlockManager&) = delete;
    KVCacheBlockManager& operator=(const KVCacheBlockManager&) = delete;

    // Allow move
    KVCacheBlockManager(KVCacheBlockManager&&) = default;
    KVCacheBlockManager& operator=(KVCacheBlockManager&&) = default;

    /**
     * @brief Allocate a new block from the free pool
     *
     * The returned block has refcount = 1 (the caller owns one reference).
     * Tensor memory is allocated lazily on first allocate of a given block id.
     *
     * @return Block ID if successful, std::nullopt if no free blocks available
     */
    std::optional<uint32_t> allocate_block();

    /**
     * @brief Add a reference to an already-allocated block
     *
     * Increments the block's refcount. Use when another owner (e.g. a prefix
     * cache entry, a second infer request sharing a prefix) needs to keep
     * the block alive alongside the original allocator.
     *
     * @param block_id Block ID returned by a prior allocate_block()
     * @throws ov::Exception if the block id is invalid or the block is free
     */
    void acquire(uint32_t block_id);

    /**
     * @brief Drop a reference to a block
     *
     * Decrements the block's refcount. When the count reaches zero the
     * block is returned to the free pool (its tensor memory is retained
     * for reuse on the next allocate of the same id).
     *
     * @param block_id Block ID to release
     * @throws ov::Exception if the block id is invalid or already free
     */
    void release(uint32_t block_id);

    /**
     * @brief Current reference count for a block
     *
     * @param block_id Block ID
     * @return Reference count (0 = free, >0 = allocated)
     */
    uint32_t refcount(uint32_t block_id) const;

    /**
     * @brief Get the tensor associated with a block
     *
     * @param block_id Block ID
     * @return Tensor for the block
     */
    ov::SoPtr<ov::ITensor> get_block_tensor(uint32_t block_id) const;

    /**
     * @brief Update the number of tokens stored in a block
     *
     * @param block_id Block ID
     * @param num_tokens New token count (must be <= block_size)
     */
    void update_block_tokens(uint32_t block_id, uint32_t num_tokens);

    /**
     * @brief Get the number of tokens in a block
     *
     * @param block_id Block ID
     * @return Number of tokens
     */
    uint32_t get_block_tokens(uint32_t block_id) const;

    /**
     * @brief Get list of all currently allocated block IDs
     *
     * @return Vector of block IDs
     */
    std::vector<uint32_t> get_allocated_blocks() const;

    /**
     * @brief Forcibly reset all blocks to FREE state and clear token counts.
     *
     * Resets every block's refcount to 0 regardless of outstanding references
     * and clears token counts. Tensor memory is retained in the pool for
     * reuse; no device deallocation occurs.
     *
     * If any block had a non-zero refcount at the time of the call a warning
     * is logged — outstanding refs likely indicate a missing release() in
     * caller code. Use clear_all() for end-of-conversation/end-of-test
     * teardown; use release() for per-block lifecycle.
     */
    void clear_all();

    /**
     * @brief Get block size (tokens per block)
     */
    uint32_t get_block_size() const {
        return block_size_;
    }

    /**
     * @brief Get maximum number of blocks
     */
    uint32_t get_max_blocks() const {
        return max_blocks_;
    }

    /**
     * @brief Get number of currently free (unallocated) blocks
     */
    uint32_t num_free_blocks() const {
        return static_cast<uint32_t>(free_block_ids_.size());
    }

private:
    /**
     * @brief Represents a single block of KV cache memory
     */
    struct Block {
        ov::SoPtr<ov::ITensor> tensor;  ///< Block memory tensor (allocated on-demand)
        uint32_t num_tokens = 0;        ///< Number of tokens stored in this block
        uint32_t refcount = 0;          ///< Reference count; >0 = allocated, 0 = free
    };

    uint32_t block_size_;                  ///< Number of tokens per block
    uint32_t max_blocks_;                  ///< Maximum blocks in pool
    std::vector<Block> blocks_;            ///< All blocks (free + allocated)
    std::stack<uint32_t> free_block_ids_;  ///< Stack of free block IDs (LIFO for better reuse)

    ov::element::Type element_type_;             ///< Element type for tensors
    ov::Shape block_shape_;                      ///< Shape for block tensors
    std::string device_;                         ///< Target device
    std::shared_ptr<const ov::IPlugin> plugin_;  ///< Plugin for memory allocation

    /**
     * @brief Validate block ID
     */
    void validate_block_id(uint32_t block_id) const;
};

}  // namespace npuw
}  // namespace ov
