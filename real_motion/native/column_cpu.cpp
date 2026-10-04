// Header-free C ABI: no Python/Torch/CUDA dependency, no floating-point math.
// Buffers belong to the caller; no global scratch, internal threads or RNG.
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
using i64 = long long;
using i32 = int;
using u8 = unsigned char;
using u64 = unsigned long long;
using u32 = unsigned int;
static_assert(sizeof(i64) == 8 && sizeof(i32) == 4 && sizeof(u8) == 1, "unsupported ABI");

#ifdef _WIN32
// This integer-only library needs neither the MSVC CRT nor a Windows SDK.
// Volatile loops prevent the compiler from turning these implementations into
// recursive calls to the very intrinsics they implement. Linux uses libc.
extern "C" void* memset(void*, int, decltype(sizeof(0)));
extern "C" void* memcpy(void*, const void*, decltype(sizeof(0)));
#pragma function(memset, memcpy)
extern "C" void* memset(void* dst, int value, decltype(sizeof(0)) n) {
    volatile u8* p = static_cast<volatile u8*>(dst);
    for (decltype(n) i = 0; i < n; ++i) p[i] = static_cast<u8>(value);
    return dst;
}
extern "C" void* memcpy(void* dst, const void* src, decltype(sizeof(0)) n) {
    volatile u8* p = static_cast<volatile u8*>(dst);
    const volatile u8* q = static_cast<const volatile u8*>(src);
    for (decltype(n) i = 0; i < n; ++i) p[i] = q[i];
    return dst;
}
#endif

API int swfm_column_cpu_abi() noexcept { return 3; }

// Original six TRAIN strata, each retaining ascending candidate-row order.
// Counts/fill are integer-only; random draws ALWAYS remain on the caller.
API int swfm_sampling_strata(const u8* kinds, const i32* actors, const u8* positive,
    i64 n, i64* rows, i64* offsets) noexcept {
    if (n < 0) return -1;
    i64 counts[6] = {0, 0, 0, 0, 0, 0};
    for (i64 row = 0; row < n; ++row) {
        if (kinds[row] > 1 || positive[row] > 1) return -2;
        const int population = kinds[row] == 0 ? 0 : actors[row] < 0 ? 1 : 2;
        ++counts[population*2+(positive[row] ? 0 : 1)];
    }
    offsets[0] = 0;
    i64 cursor[6];
    for (int b = 0; b < 6; ++b) {
        offsets[b+1] = offsets[b]+counts[b]; cursor[b] = offsets[b];
    }
    for (i64 row = 0; row < n; ++row) {
        const int population = kinds[row] == 0 ? 0 : actors[row] < 0 ? 1 : 2;
        const int b = population*2+(positive[row] ? 0 : 1);
        rows[cursor[b]++] = row;
    }
    return 0;
}

// Reject overflow BEFORE any multiplication/indexing.
static i64 cells(i64 x, i64 y, i64 z) noexcept {
    constexpr i64 limit = 0x7fffffffffffffffLL;
    if (x <= 0 || y <= 0 || z <= 0 || x > limit/y || x*y > limit/z) return -1;
    return x*y*z;
}

API int swfm_rows(const i64* xy, const u8* classes, const u8* allowed,
    const u8* baseline, const i32* owners, const u8* restored, i64 n,
    i64 xsize, i64 ysize, i64 zsize, i32 kind, i32 actor,
    i64* flat, u8* base, u8* fallback, u8* legal, u8* active) noexcept {
    if (n < 0 || cells(xsize, ysize, zsize) < 0 || (kind != 0 && kind != 1)) return -1;
    for (i64 row = 0; row < n; ++row) {
        const i64 x = xy[row*2], y = xy[row*2+1];
        if (x < 0 || x >= xsize || y < 0 || y >= ysize || classes[row] >= 17) return -2;
        const i64 start = (x*ysize+y)*zsize;
        u8 changes = 0;
        for (i64 z = 0; z < zsize; ++z) {
            const i64 out = row*zsize+z, at = start+z;
            const u8 before = baseline[at];
            u8 after = before, remove = 0;
            const u8 add = before == 17 && allowed[out] != 0;
            if (kind == 1) {
                const bool own = actor >= 0 ? owners[at] == actor : before == classes[row];
                if (own) after = actor >= 0 ? restored[at] : 17;
                remove = own && before == classes[row] && after != before;
            }
            flat[out] = at; base[out] = before; fallback[out] = after;
            legal[out*3] = 1; legal[out*3+1] = add; legal[out*3+2] = remove;
            changes |= add | remove;
        }
        active[row] = changes;
    }
    return 0;
}

API int swfm_targets(const i64* flat, const u8* classes, const u8* base,
    const u8* fallback, const u8* legal, const u8* gt, i64 n, i64 zsize,
    i64 gt_size, i64* labels) noexcept {
    if (n < 0 || zsize <= 0 || gt_size < 0 || cells(1, n+1, zsize) < 0) return -1;
    for (i64 row = 0; row < n; ++row) {
        for (i64 z = 0; z < zsize; ++z) {
            const i64 out = row*zsize+z, at = flat[out];
            if (at < 0 || at >= gt_size) return -2;
            const u8 value = gt[at];
            i64 target = 0;
            if (legal[out*3+1] && value == classes[row]) target = 1;
            if (legal[out*3+2] && value == fallback[out] && value != base[out]) target = 2;
            labels[out] = target;
        }
    }
    return 0;
}

API int swfm_generation(const i64* potential, const u8* baseline, i64 n,
    i64 xsize, i64 ysize, i64 zsize, u8* active) noexcept {
    if (n < 0 || cells(xsize, ysize, zsize) < 0) return -1;
    for (i64 row = 0; row < n; ++row) {
        const i64 x = potential[row*2], y = potential[row*2+1];
        if (x < 0 || x >= xsize || y < 0 || y >= ysize) return -2;
        const i64 start = (x*ysize+y)*zsize;
        u8 any = 0;
        for (i64 z = 0; z < zsize; ++z) if (baseline[start+z] == 17) { any = 1; break; }
        active[row] = any;
    }
    return 0;
}

API i64 swfm_static(const u8* footprint, const u8* historical, const u8* memory,
    const u8* baseline, i64 xsize, i64 ysize, i64 zsize, i64* xy) noexcept {
    if (cells(xsize, ysize, zsize) < 0) return -1;
    i64 count = 0;
    for (i64 at = 0; at < xsize*ysize; ++at) {
        if (!footprint[at]) continue;
        const i64 start = at*zsize;
        bool mismatch = false;
        for (i64 z = 0; z < zsize; ++z) {
            if (historical[start+z] && memory[start+z] != baseline[start+z]) { mismatch = true; break; }
        }
        if (mismatch) { xy[count*2] = at/ysize; xy[count*2+1] = at%ysize; ++count; }
    }
    return count;
}

// Visit only the immutable history-positive XY population. No GT, learned
// pose or sampled subset defines this list; output retains its original order.
API int swfm_static_roi(const i32* xy, i64 n, const u8* historical,
    const u8* memory, const u8* baseline, i64 xsize, i64 ysize, i64 zsize,
    u8* active) noexcept {
    if (n < 0 || cells(xsize, ysize, zsize) < 0) return -1;
    for (i64 row = 0; row < n; ++row) {
        const i64 x = xy[row*2], y = xy[row*2+1];
        if (x < 0 || x >= xsize || y < 0 || y >= ysize) return -2;
        const i64 start = (x*ysize+y)*zsize;
        active[row] = 0;
        for (i64 z = 0; z < zsize; ++z) {
            if (historical[start+z] && memory[start+z] != baseline[start+z]) {
                active[row] = 1; break;
            }
        }
    }
    return 0;
}

API i64 swfm_support(const i64* flat, i64 n, i64 xsize, i64 ysize, i64 zsize,
    u8* scratch, i64* xy, i64* zbounds) noexcept {
    const i64 total = cells(xsize, ysize, zsize);
    if (n < 0 || total < 0) return -1;
    zbounds[0] = zsize; zbounds[1] = -1;
    if (!n) return 0;
    i64 minx = xsize, maxx = -1, miny = ysize, maxy = -1;
    for (i64 at = 0; at < xsize*ysize; ++at) scratch[at] = 0;
    for (i64 j = 0; j < n; ++j) {
        const i64 at = flat[j];
        if (at < 0 || at >= total) return -2;
        const i64 column = at/zsize, z = at%zsize, x = column/ysize, y = column%ysize;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < zbounds[0]) zbounds[0] = z; if (z > zbounds[1]) zbounds[1] = z;
        scratch[column] = 1;
        if (x) scratch[column-ysize] = 1;
        if (x+1 < xsize) scratch[column+ysize] = 1;
        if (y) scratch[column-1] = 1;
        if (y+1 < ysize) scratch[column+1] = 1;
    }
    if (minx) --minx; if (maxx+1 < xsize) ++maxx;
    if (miny) --miny; if (maxy+1 < ysize) ++maxy;
    i64 count = 0;
    for (i64 x = minx; x <= maxx; ++x) for (i64 y = miny; y <= maxy; ++y) {
        if (scratch[x*ysize+y]) { xy[count*2] = x; xy[count*2+1] = y; ++count; }
    }
    return count;
}

static bool contains(const i64* owned, i64 n, i64 value) noexcept {
    i64 lo = 0, hi = n;
    while (lo < hi) { const i64 mid = lo+(hi-lo)/2; if (owned[mid] < value) lo = mid+1; else hi = mid; }
    return lo < n && owned[lo] == value;
}

API int swfm_gather(const i64* indices, i64 n, const u8* history, const u8* observed,
    i64 xsize, i64 ysize, i64 zsize, i32 has_owner, const i64* owned, i64 owned_n,
    const u8* table, i64 table_start, i64 table_n, u8* labels, u8* flags) noexcept {
    if (n < 0 || owned_n < 0 || table_n < 0 || cells(xsize, ysize, zsize) < 0) return -1;
    for (i64 j = 0; j < n; ++j) {
        labels[j] = 18; flags[j] = 0;
        const i64 x = indices[j*3], y = indices[j*3+1], z = indices[j*3+2];
        if (x < 0 || x >= xsize || y < 0 || y >= ysize || z < 0 || z >= zsize) continue;
        const i64 at = (x*ysize+y)*zsize+z;
        labels[j] = history[at]; u8 bits = observed[at];
        if (has_owner) {
            const bool member = table_n ? at >= table_start && at-table_start < table_n && table[at-table_start]
                                        : contains(owned, owned_n, at);
            if (member) bits |= 2;
        }
        flags[j] = bits;
    }
    return 0;
}

API int swfm_expand(const u8* history, const u8* flags, const i64* inverse,
    const u8* classes, i64 n, i64 unique_n, i64 row_size, i32 static_actor,
    u8* out_history, u8* out_flags) noexcept {
    if (n < 0 || unique_n < 0 || row_size <= 0 || cells(1, n+1, row_size) < 0) return -1;
    for (i64 row = 0; row < n; ++row) {
        const i64 from = inverse[row];
        if (from < 0 || from >= unique_n) return -2;
        for (i64 j = 0; j < row_size; ++j) {
            const u8 value = history[from*row_size+j];
            out_history[row*row_size+j] = value;
            out_flags[row*row_size+j] = flags[from*row_size+j] | (static_actor && value == classes[row] ? 2 : 0);
        }
    }
    return 0;
}

API int swfm_changed(const i64* labels, i64 n, i64 zsize, u8* changed) noexcept {
    if (n < 0 || zsize <= 0 || cells(1, n+1, zsize) < 0) return -1;
    for (i64 row = 0; row < n; ++row) {
        u8 any = 0;
        for (i64 z = 0; z < zsize; ++z) if (labels[row*zsize+z] != 0) { any = 1; break; }
        changed[row] = any;
    }
    return 0;
}

// TRAIN scan mode never allocates dense voxel rows. Materialize mode is used
// only AFTER the original complete-population RNG draw. Both share this logic.
API int swfm_compact_columns(const i32* xy, const u8* kinds, const i32* actors,
    const u8* classes, const u64* masks, const u8* baseline, const i32* owners,
    const u8* restored, const u8* gt, i64 n, i64 xs, i64 ys, i64 zs, i32 mode,
    u8* active, u8* positive, i64* flat, u8* base, u8* fallback, u8* legal, i64* labels,
    i64* prior_counts) noexcept {
    if (n < 0 || cells(xs, ys, zs) < 0 || zs > 64 || mode < 0 || mode > 2) return -1;
    if (mode == 2) for (i64 i = 0; i < 5; ++i) prior_counts[i] = 0;
    for (i64 row = 0; row < n; ++row) {
        const i64 x = xy[row*2], y = xy[row*2+1]; const i32 actor = actors[row];
        const u8 kind = kinds[row], cls = classes[row];
        if (x < 0 || x >= xs || y < 0 || y >= ys || cls >= 17 || kind > 1
            || (kind == 0 && actor != -3) || (kind == 1 && actor < -2)
            || (zs < 64 && (masks[row] >> zs))) return -2;
        u8 changes = 0, changed = 0; const i64 start = (x*ys+y)*zs;
        for (i64 z = 0; z < zs; ++z) {
            const i64 at = start+z; const u8 before = baseline[at];
            u8 after = before, remove = 0;
            const u8 add = before == 17 && ((masks[row] >> z)&1);
            if (kind == 1) {
                const bool own = actor >= 0 ? owners[at] == actor : before == cls;
                if (own) after = actor >= 0 ? restored[at] : 17;
                remove = own && before == cls && after != before;
            }
            if (before > 17 || after > 17 || gt[at] > 17) return -3;
            i64 target = 0;
            if (add && gt[at] == cls) target = 1;
            if (remove && gt[at] == after && gt[at] != before) target = 2;
            changes |= add|remove; changed |= target != 0;
            if (mode == 2) {
                if (kind == 0 && add) ++prior_counts[target == 1 ? 1 : 0];
                if (kind == 1 && (add || remove)) ++prior_counts[2+target];
            }
            if (mode == 1) {
                const i64 out = row*zs+z;
                flat[out] = at; base[out] = before; fallback[out] = after;
                legal[out*3] = 1; legal[out*3+1] = add; legal[out*3+2] = remove; labels[out] = target;
            }
        }
        active[row] = changes; positive[row] = changed;
    }
    return 0;
}

// One source-list call per horizon. Stamps avoid clearing the entire BEV grid
// for each source; results retain full-grid argwhere lexicographic order.
API i64 swfm_support_many(const i64* flat, const i64* offsets, i64 groups,
    i64 total_points, i64 xs, i64 ys, i64 zs, u32* stamp, i64 capacity,
    i64* xy, i64* row_offsets, i64* zbounds) noexcept {
    const i64 total = cells(xs, ys, zs);
    if (groups < 0 || groups >= 0xffffffffLL || total < 0 || total_points < 0 || capacity < 0
        || offsets[0] != 0 || offsets[groups] != total_points) return -1;
    for (i64 i = 0; i < xs*ys; ++i) stamp[i] = 0;
    i64 count = 0; row_offsets[0] = 0;
    for (i64 group = 0; group < groups; ++group) {
        const i64 begin = offsets[group], end = offsets[group+1];
        if (begin < 0 || begin > end || end > total_points) return -2;
        const u32 mark = static_cast<u32>(group+1);
        i64 minx = xs, maxx = -1, miny = ys, maxy = -1;
        zbounds[group*2] = zs; zbounds[group*2+1] = -1;
        for (i64 j = begin; j < end; ++j) {
            const i64 at = flat[j]; if (at < 0 || at >= total) return -2;
            const i64 column = at/zs, z = at%zs, x = column/ys, y = column%ys;
            if (x < minx) minx = x; if (x > maxx) maxx = x;
            if (y < miny) miny = y; if (y > maxy) maxy = y;
            if (z < zbounds[group*2]) zbounds[group*2] = z;
            if (z > zbounds[group*2+1]) zbounds[group*2+1] = z;
            stamp[column] = mark;
            if (x) stamp[column-ys] = mark; if (x+1 < xs) stamp[column+ys] = mark;
            if (y) stamp[column-1] = mark; if (y+1 < ys) stamp[column+1] = mark;
        }
        if (begin != end) {
            if (minx) --minx; if (maxx+1 < xs) ++maxx;
            if (miny) --miny; if (maxy+1 < ys) ++maxy;
            for (i64 x = minx; x <= maxx; ++x) for (i64 y = miny; y <= maxy; ++y) {
                if (stamp[x*ys+y] == mark) {
                    if (count >= capacity) return -4;
                    xy[count*2] = x; xy[count*2+1] = y; ++count;
                }
            }
        }
        row_offsets[group+1] = count;
    }
    return count;
}

// Integer gathering for every historical frame, writing directly into the
// original [anchor,frame,patch_voxel] layout (no frame-transpose copies).
API int swfm_gather_many(const i64* const* indices, i64 frames, i64 points, i64 row_size,
    const u8* history, const u8* observed, i64 xs, i64 ys, i64 zs,
    const u8* valid_frame, const u8* has_owner, const i64* const* owned, const i64* owned_n,
    const u8* const* table, const i64* table_start, const i64* table_n,
    u8* labels, u8* flags) noexcept {
    const i64 total = cells(xs, ys, zs);
    if (total < 0 || frames <= 0 || points < 0 || row_size <= 0 || points%row_size
        || cells(1, frames, points+1) < 0) return -1;
    for (i64 f = 0; f < frames; ++f) {
        if (owned_n[f] < 0 || table_n[f] < 0 || table_start[f] < 0) return -1;
        for (i64 block = 0; block < points; block += row_size) {
          const i64 out_start = (block/row_size*frames+f)*row_size;
          for (i64 local = 0; local < row_size; ++local) {
            const i64 j = block+local, out = out_start+local;
            labels[out] = 18; flags[out] = 0;
            if (!valid_frame[f]) continue;
            const i64 in = j*3;
            const i64 x = indices[f][in], y = indices[f][in+1], z = indices[f][in+2];
            if (x < 0 || x >= xs || y < 0 || y >= ys || z < 0 || z >= zs) continue;
            const i64 at = (x*ys+y)*zs+z;
            labels[out] = history[f*total+at]; u8 bits = observed[f*total+at];
            if (has_owner[f]) {
                const bool member = table_n[f] ? at >= table_start[f] && at-table_start[f] < table_n[f] && table[f][at-table_start[f]]
                                              : contains(owned[f], owned_n[f], at);
                if (member) bits |= 2;
            }
            flags[out] = bits;
          }
        }
    }
    return 0;
}
