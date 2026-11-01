#pragma once
// Prompt sweeps (engine/decoder.cuh): a chunk of Rounds goes through the model layer by layer, so each
// Round's state lives in a slot of a chunk array between its layer steps. These copies move one
// Round's slot -- the Round chosen by a device value, so the captured layer graphs replay for every
// Round -- in and out of the working buffers the layer step reads.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ds4n::cuda {

/** dst[0 .. bytes) = base[*round * slot_bytes ..]: Round *round's slot into a working buffer. */
void load_slot(void* dst, const void* base, size_t slot_bytes, const int32_t* round, size_t bytes, cudaStream_t stream);

/** base[*round * slot_bytes ..] = src[0 .. bytes): a working buffer into Round *round's slot. */
void store_slot(void* base, size_t slot_bytes, const int32_t* round, const void* src, size_t bytes, cudaStream_t stream);

/**
 * The streams a chunk's Rounds start from: Round g's row r is token 8g + r, its embedding
 * (`embeddings` [tokens][dim] BF16) in all four hc copies of h ([rounds][8][4][dim]).
 */
void expand_embeddings(const __nv_bfloat16* embeddings, int tokens, int dim, __nv_bfloat16* h, cudaStream_t stream);

}  // namespace ds4n::cuda
