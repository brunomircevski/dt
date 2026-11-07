#pragma once

#include "algo/split_math.h"
#include "core/dataset.h"
#include "core/thread_pool.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dt {

// The slot of every feature's column (see Columns in cpu_builder.h).
using SlotMap = std::vector<std::uint32_t>;

// Pick the packing of (row, class) into 32 bits for this dataset.
EntryCodec makeEntryCodec(std::size_t classCount, std::size_t rowCount);

// Sort every feature column by value (LSD radix sort, stable, so equal
// values keep row order) into `slots` slots of rowCount entries at `data`
// (at least featureCount + 1 of them): the sorts use the slots that hold no
// sorted column as workspace. Returns the slot of every feature.
SlotMap presortColumns(const Dataset &dataset, const EntryCodec &codec, Entry *data,
                       std::size_t slots, ThreadPool *pool);

} // namespace dt
