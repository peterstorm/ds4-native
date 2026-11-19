#pragma once
// Imperative shell: GLM-5.3 Backbone parts on disk. The plan (which tensors,
// where, the exact header bytes) is core/glm53/backbone; this file only reads
// source shards and writes / compares part files.
//
// Writing is retry-safe: a part is written to "<name>.partial", fsynced and
// renamed into place, so a part file under its final name is always complete.
// An existing final part is kept only when it verifies; a damaged one is
// rewritten and replaced the same way.
#include <cstddef>
#include <cstdint>
#include <string>

#include "core/glm53/backbone.hpp"
#include "core/result.hpp"
#include "io/file.hpp"

namespace ds4n::io {

/** Read a source shard's header and plan its Backbone part. */
Result<glm53::BackbonePart, IoError> plan_backbone_part(const std::string& shard_path);

enum class Extraction { Written, Repaired, AlreadyPresent, NoBackbone };

/** Write `plan` (from plan_backbone_part(shard_path)) into `out_dir`. */
Result<Extraction, IoError> write_backbone_part(const std::string& shard_path, const glm53::BackbonePart& plan,
                                               const std::string& out_dir);

/**
 * Proof that the shard's Backbone is saved: its part file in `out_dir` equals
 * plan_part(shard) byte for byte — size, header and every tensor — or the
 * shard has no Backbone tensor and needs no part (`tensors` 0, `path` empty).
 * Any difference, a missing file or a short one is an error.
 */
struct VerifiedPart {
  std::string path;
  size_t tensors;
  uint64_t bytes;
};
Result<VerifiedPart, IoError> verify_backbone_part(const std::string& shard_path, const std::string& out_dir);

/**
 * Refuse a write of `bytes` into `dir` unless the filesystem keeps
 * `floor_bytes` free afterwards. rpool is shared with the serving stack and
 * ENOSPC there takes it down, so both GLM-5.3 converters check before writing.
 */
Result<bool, IoError> ensure_headroom(const std::string& dir, uint64_t bytes, uint64_t floor_bytes);

}  // namespace ds4n::io
