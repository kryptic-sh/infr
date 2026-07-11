// Shared native-block dequant library (raw GGUF blocks, padded to u32). Included by native_gemv
// (decode GEMV) and native_gemm (prefill tiled coopmat GEMM). The includer must declare the weight
// SSBO as `uint nw[]` (any binding). A -DFMT_* define selects one format. Two interfaces per format:
//   * dq(uint g)                      — single element g (global index into [N,K] = out*K + in)
//   * dqblk(uint gstart, out v[32])   — decode a contiguous 32-elem sub-block, scale decoded ONCE.
// dqblk is the amortized path; formats without an optimized one fall back to looping dq() (below).
// Grid-based i-quants pull their tables from native_grids.glsl.
//
// NW(i): the nw[] read chokepoint. An includer may pre-define NW to add a WORD-offset base to
// every weight read — the paged expert kernels (`native_gemv_id(_multi).comp`'s -DPAGED build) use
// this to relocate all dequant math into one arena slot: the within-expert ELEMENT offsets this
// library computes stay small (they fit u32 comfortably for any real expert), while the slot's
// arena position — whose ELEMENT/BYTE offset would overflow u32 past a few dozen ~14 MB slots
// (u32 element math wrapped at slot ≥ ~102 on Scout, the task's coherent-but-wrong bug) — is
// carried as a u32 WORD index added only here, at the final indexing step (a 16 GiB-per-arena
// reach, enforced host-side; see `GpuPager`). Undefined = plain nw[i]: every non-paged build
// expands to exactly the code this library always had.
#ifndef NW
#define NW(i) nw[i]
#endif

uint rb(uint bo) { return (NW(bo >> 2u) >> ((bo & 3u) << 3u)) & 0xFFu; }
uint ru16(uint bo) { return rb(bo) | (rb(bo + 1u) << 8u); }
uint ru32b(uint bo) { return rb(bo) | (rb(bo + 1u) << 8u) | (rb(bo + 2u) << 16u) | (rb(bo + 3u) << 24u); }
// Unaligned u32 via two word loads + funnel shift (vs ru32b's four byte-extract chains).
uint ru32u(uint bo) {
    uint w0 = NW(bo >> 2u);
    uint sh = (bo & 3u) << 3u;
    if (sh == 0u) { return w0; }
    return (w0 >> sh) | (NW((bo >> 2u) + 1u) << (32u - sh));
}
float f16tof32(uint bits) { return unpackHalf2x16(bits & 0xffffu).x; }
int sgn8(uint byte) { return int(byte) - int(byte >= 128u ? 256u : 0u); }

// USE_GRID builds stage their codebook tables into `shared` memory once per workgroup
// (grid_init() in the generated native_grids.glsl — see build.rs's gen_grids for why const-array
// indexing is a per-invocation-scratch catastrophe on RADV/ACO). Every includer runs `GRID_INIT;`
// at the TOP of main(): it contains a barrier(), so it must precede any early return / divergent
// flow. Non-grid builds expand it to an empty statement — byte-identical SPIR-V.
#ifdef USE_GRID
#include "native_grids.glsl"
#define GRID_INIT grid_init()
#else
#define GRID_INIT
#endif

#if defined(FMT_Q8_0)
// Q8_0: [f16 d][i8 qs[32]] = 34 bytes, 32 elements. y = d * qs[j].
float dq(uint g) {
    uint bd = (g / 32u) * 34u;
    return f16tof32(ru16(bd)) * float(sgn8(rb(bd + 2u + g % 32u)));
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 34u;
    float d = f16tof32(ru16(bd));
    for (uint w = 0u; w < 32u; w++) { v[w] = d * float(sgn8(rb(bd + 2u + w))); }
}
#endif

#if defined(FMT_F16)
// F16: contiguous IEEE half (no blocks/scale). y = unpackHalf2x16 — EXACT. Element e lives at
// byte e*2 (packed 2-per-u32). Serves float weights that are UPLOADED as f16 (e.g. the
// DiffusionGemma SC soft-embedding, host-built f16 [ne, vocab]) when a warptile shape needs the
// native GEMM machinery (split-K) that the plain f16 `gemm_proj` kernel lacks.
float dq(uint g) {
    return f16tof32(ru16(g * 2u));
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bo = gstart * 2u;
    for (uint w = 0u; w < 32u; w++) { v[w] = f16tof32(ru16(bo + w * 2u)); }
}
#endif

#if defined(FMT_F32)
// F32: contiguous IEEE single (no blocks/scale) — one weight element per u32 word, EXACT. Serves
// the PAGED MoE expert path only: a paged float expert bank stages its RAW GGUF bytes into the
// arena (no host f16 conversion — resident float banks go through `bind_weight`, which converts
// to f16 and reports the effective dtype, so they hit FMT_F16 instead).
float dq(uint g) {
    return uintBitsToFloat(NW(g));
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    for (uint w = 0u; w < 32u; w++) { v[w] = uintBitsToFloat(NW(gstart + w)); }
}
#endif

#if defined(FMT_BF16)
// BF16: contiguous 16-bit truncated-f32 (no blocks/scale). y = bitcast(bits << 16) — EXACT (bf16 is
// the top 16 bits of an f32). Element e lives at byte e*2 (packed 2-per-u32).
float dq(uint g) {
    return uintBitsToFloat(ru16(g * 2u) << 16u);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bo = gstart * 2u;
    for (uint w = 0u; w < 32u; w++) { v[w] = uintBitsToFloat(ru16(bo + w * 2u) << 16u); }
}
#endif

#if defined(FMT_Q4_0)
// Q4_0: [f16 d][u8 qs[16]] = 18 bytes, 32 elements. y = d * (nibble - 8).
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    uint nib = (j < 16u) ? (rb(bd + 2u + j) & 0xFu) : (rb(bd + 2u + j - 16u) >> 4u);
    return d * (float(nib) - 8.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    for (uint w = 0u; w < 32u; w++) {
        uint nib = (w < 16u) ? (rb(bd + 2u + w) & 0xFu) : (rb(bd + 2u + w - 16u) >> 4u);
        v[w] = d * (float(nib) - 8.0);
    }
}
#endif

#if defined(FMT_Q4_1)
// Q4_1: [f16 d][f16 m][u8 qs[16]] = 20 bytes, 32 elements. y = d*nibble + m.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 20u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u));
    uint nib = (j < 16u) ? (rb(bd + 4u + j) & 0xFu) : (rb(bd + 4u + j - 16u) >> 4u);
    return d * float(nib) + m;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 20u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u));
    for (uint w = 0u; w < 32u; w++) {
        uint nib = (w < 16u) ? (rb(bd + 4u + w) & 0xFu) : (rb(bd + 4u + w - 16u) >> 4u);
        v[w] = d * float(nib) + m;
    }
}
#endif

#if defined(FMT_Q5_0)
// Q5_0: [f16 d][u8 qh[4]][u8 qs[16]] = 22 bytes, 32 elements. y = d*(q5 - 16).
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 22u;
    float d = f16tof32(ru16(bd)); uint qh = ru32b(bd + 2u); uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (rb(bd + 6u + j) & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 6u + jj) >> 4u) | xh1; }
    return d * (float(val) - 16.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 22u;
    float d = f16tof32(ru16(bd)); uint qh = ru32b(bd + 2u);
    for (uint w = 0u; w < 32u; w++) {
        uint val;
        if (w < 16u) { uint xh0 = ((qh >> w) << 4u) & 0x10u; val = (rb(bd + 6u + w) & 0xFu) | xh0; }
        else { uint jj = w - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 6u + jj) >> 4u) | xh1; }
        v[w] = d * (float(val) - 16.0);
    }
}
#endif

#if defined(FMT_Q5_1)
// Q5_1: [f16 d][f16 m][u8 qh[4]][u8 qs[16]] = 24 bytes, 32 elements. y = d*q5 + m.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 24u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u)); uint qh = ru32b(bd + 4u); uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (rb(bd + 8u + j) & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 8u + jj) >> 4u) | xh1; }
    return d * float(val) + m;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 24u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u)); uint qh = ru32b(bd + 4u);
    for (uint w = 0u; w < 32u; w++) {
        uint val;
        if (w < 16u) { uint xh0 = ((qh >> w) << 4u) & 0x10u; val = (rb(bd + 8u + w) & 0xFu) | xh0; }
        else { uint jj = w - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 8u + jj) >> 4u) | xh1; }
        v[w] = d * float(val) + m;
    }
}
#endif

#if defined(FMT_Q2K)
// Q2_K: [u8 scales[16]][u8 qs[64]][f16 d][f16 dmin] = 84 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 84u;
    float d = f16tof32(ru16(bd + 80u)); float dmin = f16tof32(ru16(bd + 82u));
    uint sc_byte = rb(bd + p / 16u);
    float dl = d * float(sc_byte & 0xFu); float ml = dmin * float(sc_byte >> 4u);
    uint n = p / 128u; uint p_half = p % 128u; uint jj = p_half / 32u; uint p_j = p_half % 32u;
    uint q2 = (rb(bd + 16u + 32u * n + p_j) >> (2u * jj)) & 3u;
    return dl * float(q2) - ml;
}
#define HAVE_DQBLK
// A 32-aligned run has CONSTANT n and jj, so its 2-bit quants are 32 CONSECUTIVE bytes = 8 aligned
// u32 words, under just 2 scale bytes — load those 10 values once instead of 2 byte-extract chains
// per element (the naive form was ~64 word loads per sub-block; the measured 14B-Q2_K decode
// catastrophe, 0.26x llama).
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 84u; uint p0 = gstart % 256u;
    float d = f16tof32(ru16(bd + 80u)); float dmin = f16tof32(ru16(bd + 82u));
    uint sb0 = rb(bd + p0 / 16u);        // scale/min nibbles for elements 0..15
    uint sb1 = rb(bd + p0 / 16u + 1u);   // … and 16..31
    float dl0 = d * float(sb0 & 0xFu); float ml0 = dmin * float(sb0 >> 4u);
    float dl1 = d * float(sb1 & 0xFu); float ml1 = dmin * float(sb1 >> 4u);
    uint shift = 2u * ((p0 % 128u) / 32u);
    uint qw = (bd + 16u + 32u * (p0 / 128u)) >> 2u; // 84-byte blocks are word-aligned; +16+32n too
    for (uint w8 = 0u; w8 < 8u; w8++) {
        // Hoist the plane shift to a single packed op: one shift+mask isolates all four 2-bit
        // codes, then a per-element bitfieldExtract (1 op) replaces the >>8b >>shift &3 chain.
        // Same integers, same v[w] expression — bit-identical to the per-element form.
        uint q4 = (NW(qw + w8) >> shift) & 0x03030303u;
        for (uint b = 0u; b < 4u; b++) {
            uint w = w8 * 4u + b;
            uint q2 = bitfieldExtract(q4, int(8u * b), 2);
            v[w] = (w < 16u) ? (dl0 * float(q2) - ml0) : (dl1 * float(q2) - ml1);
        }
    }
}
#endif

#if defined(FMT_Q3K)
// Q3_K: [u8 hmask[32]][u8 qs[64]][u8 scales_raw[12]][f16 d] = 110 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 110u;
    float d_all = f16tof32(ru16(bd + 108u));
    uint a0 = ru32b(bd + 96u); uint a1 = ru32b(bd + 100u); uint a2 = ru32b(bd + 104u);
    uint k1 = 0x03030303u; uint k2 = 0x0f0f0f0fu; uint tmp = a2;
    uint aux[4];
    aux[2] = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
    aux[0] = (a0 & k2) | ((tmp & k1) << 4u);
    aux[1] = (a1 & k2) | (((tmp >> 2u) & k1) << 4u);
    uint is = p / 16u;
    uint sc_byte = (aux[is >> 2u] >> ((is & 3u) * 8u)) & 0xFFu;
    int sc = int(sc_byte) - int(sc_byte >= 128u ? 256u : 0u) - 32;
    float dl = d_all * float(sc);
    uint n = p / 128u; uint p_half = p % 128u; uint jj = p_half / 32u; uint p_j = p_half % 32u;
    uint jg = 4u * n + jj; uint m = 1u << jg;
    uint low2 = (rb(bd + 32u + 32u * n + p_j) >> (2u * jj)) & 3u;
    uint high = ((rb(bd + p_j) & m) != 0u) ? 1u : 0u;
    return dl * (float(low2 | (high << 2u)) - 4.0);
}
#define HAVE_DQBLK
// A 32-aligned run has constant n/jj, so its low-2-bit quants are 32 CONSECUTIVE bytes (8 unaligned
// u32s via ru32u — 110-byte blocks aren't word-aligned) and its high bits are one fixed mask over
// the 32 hmask bytes; only 2 of the 16 6-bit scales apply. The naive form did 2 byte-extract
// chains per ELEMENT (the o/down Q3_K GEMVs in Q2_K-mix models were the decode bottleneck).
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 110u; uint p0 = gstart % 256u;
    float d_all = f16tof32(ru16(bd + 108u));
    // ru32u (2 word loads + funnel) instead of ru32b (4 byte-extract chains) — same value.
    uint a0 = ru32u(bd + 96u); uint a1 = ru32u(bd + 100u); uint a2 = ru32u(bd + 104u);
    uint k1 = 0x03030303u; uint k2 = 0x0f0f0f0fu; uint tmp = a2;
    uint aux[4];
    aux[2] = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
    aux[0] = (a0 & k2) | ((tmp & k1) << 4u);
    aux[1] = (a1 & k2) | (((tmp >> 2u) & k1) << 4u);
    uint is0 = p0 / 16u;
    uint sb0 = (aux[is0 >> 2u] >> ((is0 & 3u) * 8u)) & 0xFFu;
    uint sb1 = (aux[(is0 + 1u) >> 2u] >> (((is0 + 1u) & 3u) * 8u)) & 0xFFu;
    float dl0 = d_all * float(sgn8(sb0) - 32);
    float dl1 = d_all * float(sgn8(sb1) - 32);
    uint n = p0 / 128u;
    uint jj = (p0 % 128u) / 32u;
    uint shift = 2u * jj;
    uint jg = 4u * n + jj;
    uint qb = bd + 32u + 32u * n;
    for (uint w8 = 0u; w8 < 8u; w8++) {
        // Packed plane extraction: one shift+mask isolates the four 2-bit codes, one
        // shift+mask+shl turns the four hmask bits into the packed +4 plane, one OR merges —
        // the per-element >>8b>>shift&3 chain and &m!=0 compare/select drop to a single
        // bitfieldExtract. XOR 0x04 flips the +4 plane so a SIGNED 3-bit extract yields q-4
        // directly ((q^4) as s3 == q-4 for q in 0..7), dropping the per-element -4.0 subtract.
        // float(int(q-4)) == float(q)-4.0 exactly (small integers) — bit-identical.
        uint q4 = ((ru32u(qb + w8 * 4u) >> shift) & 0x03030303u)
            | (((ru32u(bd + w8 * 4u) >> jg) & 0x01010101u) << 2u);
        int q4s = int(q4 ^ 0x04040404u);
        for (uint b = 0u; b < 4u; b++) {
            uint w = w8 * 4u + b;
            float dl = (w < 16u) ? dl0 : dl1;
            v[w] = dl * float(bitfieldExtract(q4s, int(8u * b), 3));
        }
    }
}
#endif

#if defined(FMT_Q4K) || defined(FMT_Q5K)
// k4: decode 6-bit scale/min for sub-block i from the 12-byte scales field at sb.
uvec2 k4(uint i, uint sb) {
    if (i < 4u) { return uvec2(rb(sb + i) & 63u, rb(sb + i + 4u) & 63u); }
    uint sc = (rb(sb + i + 4u) & 0xFu) | ((rb(sb + i - 4u) >> 6u) << 4u);
    uint mn = (rb(sb + i + 4u) >> 4u) | ((rb(sb + i) >> 6u) << 4u);
    return uvec2(sc, mn);
}
#endif

#if defined(FMT_Q4K)
// Q4_K: [f16 d][f16 dmin][u8 scales[12]][u8 qs[128]] = 144 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 144u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uint sb = bd + 4u; uint j = p / 64u; uint p_j = p % 64u; uint l = p_j % 32u;
    uvec2 k = (p_j < 32u) ? k4(2u * j, sb) : k4(2u * j + 1u, sb);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    uint qs_byte = rb(bd + 16u + j * 32u + l);
    uint val = (p_j < 32u) ? (qs_byte & 0xFu) : (qs_byte >> 4u);
    return dl * float(val) - mm;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // decode d/dmin/6-bit scale once for the sub-block.
    uint sblk = gstart / 32u; uint super = sblk / 8u; uint sub = sblk % 8u;
    uint bd = super * 144u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uvec2 k = k4(sub, bd + 4u);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    // Word-parallel qs: block byte offsets are 4-aligned (144%4==0, qs at +16, sub-row stride 32)
    // — load 8 whole u32s instead of 32 rb() byte-extract chains; nibble select hoisted (constant
    // per sub-block). The staging loop of the warptile GEMM is decode-ALU-bound, so this is
    // directly visible in prefill GEMM throughput.
    uint qw = (bd + 16u + (sub / 2u) * 32u) >> 2u;
    uint nsh = ((sub & 1u) == 0u) ? 0u : 4u;
    for (uint w8 = 0u; w8 < 8u; w8++) {
        uint q = NW(qw + w8) >> nsh;
        for (uint b = 0u; b < 4u; b++) {
            v[w8 * 4u + b] = dl * float((q >> (8u * b)) & 0xFu) - mm;
        }
    }
}
#endif

#if defined(FMT_Q5K)
// Q5_K: [f16 d][f16 dmin][u8 scales[12]][u8 qh[32]][u8 qs[128]] = 176 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 176u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uint sb = bd + 4u; uint j = p / 64u; uint p_j = p % 64u; uint l = p_j % 32u;
    uvec2 k = (p_j < 32u) ? k4(2u * j, sb) : k4(2u * j + 1u, sb);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    uint qs_byte = rb(bd + 48u + j * 32u + l);
    uint qh_byte = rb(bd + 16u + l);
    uint val;
    if (p_j < 32u) { val = (qs_byte & 0xFu) + (((qh_byte & (1u << (2u * j))) != 0u) ? 16u : 0u); }
    else { val = (qs_byte >> 4u) + (((qh_byte & (2u << (2u * j))) != 0u) ? 16u : 0u); }
    return dl * float(val) - mm;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // decode d/dmin/6-bit scale once for the sub-block.
    uint sblk = gstart / 32u; uint super = sblk / 8u; uint sub = sblk % 8u;
    uint bd = super * 176u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uvec2 k = k4(sub, bd + 4u);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    // Word-parallel qs+qh: block byte offsets are 4-aligned (176%4==0, qh at +16, qs at +48) —
    // 16 whole-u32 loads instead of 64 rb() byte-extract chains, nibble/high-bit selects hoisted
    // (constant per sub-block). Same `nib | (h<<4)` value as the scalar loop — bit-identical.
    uint j = sub / 2u;
    uint qw = (bd + 48u + j * 32u) >> 2u;
    uint hw = (bd + 16u) >> 2u;
    uint nsh = ((sub & 1u) == 0u) ? 0u : 4u;      // low/high nibble of qs
    uint hsh = 2u * j + (((sub & 1u) == 0u) ? 0u : 1u); // qh bit index for this sub-block
    for (uint w8 = 0u; w8 < 8u; w8++) {
        uint q = NW(qw + w8) >> nsh;
        uint h = NW(hw + w8) >> hsh;
        for (uint b = 0u; b < 4u; b++) {
            uint val = ((q >> (8u * b)) & 0xFu) | (((h >> (8u * b)) & 1u) << 4u);
            v[w8 * 4u + b] = dl * float(val) - mm;
        }
    }
}
#endif

#if defined(FMT_Q6K)
// Q6_K: [u8 ql[128]][u8 qh[64]][i8 scales[16]][f16 d] = 210 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 210u;
    float d = f16tof32(ru16(bd + 208u));
    uint hf = p / 128u; uint p_half = p % 128u; uint og = p_half / 32u; uint l = p_half % 32u;
    uint qlo = hf * 64u; uint qho = hf * 32u;
    uint qa = rb(bd + qlo + l); uint qb = rb(bd + qlo + l + 32u); uint qh = rb(bd + 128u + qho + l);
    uint q;
    if (og == 0u) { q = (qa & 0xFu) | ((qh & 3u) << 4u); }
    else if (og == 1u) { q = (qb & 0xFu) | (((qh >> 2u) & 3u) << 4u); }
    else if (og == 2u) { q = (qa >> 4u) | (((qh >> 4u) & 3u) << 4u); }
    else { q = (qb >> 4u) | (((qh >> 6u) & 3u) << 4u); }
    uint sc_idx = hf * 8u + l / 16u + 2u * og;
    int sc = sgn8(rb(bd + 192u + sc_idx));
    return d * float(sc) * (float(q) - 32.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // hf/og/d constant; hoist branch, one ql byte/elem.
    uint bd = (gstart / 256u) * 210u; uint p0 = gstart % 256u;
    float d = f16tof32(ru16(bd + 208u));
    uint hf = p0 / 128u; uint og = (p0 % 128u) / 32u;
    uint scbase = bd + 192u + hf * 8u + 2u * og;
    bool useB = (og & 1u) == 1u; bool high = og >= 2u; uint qshift = og * 2u;
    uint qoff = bd + hf * 64u + (useB ? 32u : 0u);  // pick qa/qb half once
    uint qhoff = bd + 128u + hf * 32u;
    // Word-parallel ql+qh (blocks are 210 bytes — only 2-aligned, so ru32u funnel-shift loads):
    // 16 u32 loads instead of 96 rb() byte-extract chains; nibble select and the two per-16-elem
    // scales hoisted. Same (d*sc)*(q-32) product order as the scalar loop — bit-identical.
    float dsc0 = d * float(sgn8(rb(scbase)));
    float dsc1 = d * float(sgn8(rb(scbase + 1u)));
    uint nsh = high ? 4u : 0u;
    for (uint w8 = 0u; w8 < 8u; w8++) {
        uint q = ru32u(qoff + w8 * 4u) >> nsh;
        uint h = ru32u(qhoff + w8 * 4u) >> qshift;
        float dsc = (w8 < 4u) ? dsc0 : dsc1;
        for (uint b = 0u; b < 4u; b++) {
            uint qq = ((q >> (8u * b)) & 0xFu) | (((h >> (8u * b)) & 3u) << 4u);
            v[w8 * 4u + b] = dsc * (float(qq) - 32.0);
        }
    }
}
#endif

// ── codebook formats (small const tables / decode fns, no grid) ──
#if defined(FMT_IQ4NL) || defined(FMT_IQ4XS)
// kvalues_iq4nl, byte-packed 4-per-u32 (all 16 signed values fit int8) instead of a 16-entry
// `int[]`: a dynamically-indexed 16-element array lowers (RADV/ACO) to a far longer select
// cascade than a 4-element one, and the intra-word extract is a single native bitfieldExtract
// (V_BFE, sign-extending) instead of a second array read — idx>>2 picks the word (4-way), idx&3
// the byte lane. This IS the ALU floor of IQ4_XS decode: bytes-vs-speed showed IQ4_XS (4.25 bpw,
// SMALLER than Q4_K's 4.5) running 1.55-2.1x SLOWER per dispatch at matched shapes — codebook-
// gather ALU, not DRAM bandwidth.
const uint KV_IQ4NL_W[4] = uint[](0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);
int kv_iq4nl(uint idx) {
    return bitfieldExtract(int(KV_IQ4NL_W[idx >> 2u]), int((idx & 3u) << 3u), 8);
}
#endif
#if defined(FMT_MXFP4) || defined(FMT_NVFP4)
const int KV_MXFP4[16] = int[](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
#endif
#if defined(FMT_TQ1_0)
const uint POW3[6] = uint[](1u, 3u, 9u, 27u, 81u, 243u);
#endif
#if defined(FMT_MXFP4)
float e8m0_half(uint x) {
    if (x < 2u) { return uintBitsToFloat(0x00200000u << x); }
    return uintBitsToFloat((x - 1u) << 23u);
}
#endif
#if defined(FMT_NVFP4)
float ue4m3(uint x) {
    if (x == 0u || x == 0x7Fu) { return 0.0; }
    uint e = (x >> 3u) & 0xFu; float man = float(x & 7u);
    float raw = (e == 0u) ? (man * exp2(-9.0)) : ((1.0 + man / 8.0) * exp2(float(int(e) - 7)));
    return raw * 0.5;
}
#endif

#if defined(FMT_IQ4NL)
// IQ4_NL: [f16 d][u8 qs[16]] = 18 bytes, 32 elements.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    uint idx = (j < 16u) ? (rb(bd + 2u + j) & 0xFu) : (rb(bd + 2u + j - 16u) >> 4u);
    return d * float(kv_iq4nl(idx));
}
#define HAVE_DQBLK
// One 32-elem sub-block = exactly one whole IQ4_NL block (gstart is 32-aligned). Decode d ONCE
// and load the 16 qs bytes as 4 (unaligned — 18-byte block stride) u32s instead of 32 rb()
// byte-extract chains; low nibble → v[0..16), high nibble → v[16..32). Same d*KV[nibble] value
// as the scalar loop — bit-identical. The warptile GEMM's staging loop is decode-ALU-bound,
// so this is directly visible in prefill GEMM throughput.
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    for (uint w8 = 0u; w8 < 4u; w8++) {
        uint q = ru32u(bd + 2u + w8 * 4u);
        for (uint b = 0u; b < 4u; b++) {
            uint byteq = (q >> (8u * b)) & 0xFFu;
            v[w8 * 4u + b] = d * float(kv_iq4nl(byteq & 0xFu));
            v[w8 * 4u + b + 16u] = d * float(kv_iq4nl(byteq >> 4u));
        }
    }
}
#endif

#if defined(FMT_IQ4XS)
// IQ4_XS: [f16 d][u16 scales_h][u8 scales_l[4]][u8 qs[128]] = 136 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 136u;
    float d = f16tof32(ru16(bd));
    uint scales_h = ru16(bd + 2u);
    uint ib = p / 32u; uint within = p % 32u;
    uint lo = (rb(bd + 4u + (ib / 2u)) >> (4u * (ib & 1u))) & 0xFu;
    uint hi = (scales_h >> (2u * ib)) & 3u;
    uint ls = lo | (hi << 4u);
    float dl = d * float(int(ls) - 32);
    uint qoff = bd + 8u + 16u * ib;
    uint idx = (within < 16u) ? (rb(qoff + within) & 0xFu) : (rb(qoff + within - 16u) >> 4u);
    return dl * float(kv_iq4nl(idx));
}
#define HAVE_DQBLK
// One 32-elem sub-block = exactly one of the 8 IQ4_XS sub-blocks (gstart is 32-aligned).
// Decode d and the 6-bit sub-block scale ONCE. Word-parallel qs: block byte offsets are 4-aligned
// (136%4==0, qs at +8, sub-block stride 16) — load the 16 qs bytes as 4 whole u32s instead of 16
// rb() byte-extract chains, with the codebook gather hoisted per byte. Same dl*KV[nibble] value as
// the scalar loop — bit-identical. The warptile GEMM's staging loop is decode-ALU-bound, so this is
// directly visible in prefill GEMM (and the decode native_gemv) throughput.
void dqblk(uint gstart, out float v[32]) {
    uint p = gstart % 256u;
    uint bd = (gstart / 256u) * 136u;
    float d = f16tof32(ru16(bd));
    uint scales_h = ru16(bd + 2u);
    uint ib = p / 32u;
    uint lo = (rb(bd + 4u + (ib / 2u)) >> (4u * (ib & 1u))) & 0xFu;
    uint hi = (scales_h >> (2u * ib)) & 3u;
    float dl = d * float(int(lo | (hi << 4u)) - 32);
    uint qw = (bd + 8u + 16u * ib) >> 2u;
    for (uint w4 = 0u; w4 < 4u; w4++) {
        uint q = NW(qw + w4);
        for (uint b = 0u; b < 4u; b++) {
            uint byte = (q >> (8u * b)) & 0xFFu;
            uint j = w4 * 4u + b;
            v[j] = dl * float(kv_iq4nl(byte & 0xFu));
            v[j + 16u] = dl * float(kv_iq4nl(byte >> 4u));
        }
    }
}
#endif

#if defined(FMT_MXFP4)
// MXFP4: [u8 e8m0][u8 qs[16]] = 17 bytes, 32 elements.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 17u;
    float d = e8m0_half(rb(bd));
    uint idx = (j < 16u) ? (rb(bd + 1u + j) & 0xFu) : (rb(bd + 1u + j - 16u) >> 4u);
    return float(KV_MXFP4[idx]) * d;
}
#endif

#if defined(FMT_NVFP4)
// NVFP4: [u8 scales[4]][u8 qs[32]] = 36 bytes, 64 elements.
float dq(uint g) {
    uint p = g % 64u; uint bd = (g / 64u) * 36u;
    uint s = p / 16u; uint within = p % 16u;
    float d = ue4m3(rb(bd + s));
    uint idx = (within < 8u) ? (rb(bd + 4u + s * 8u + within) & 0xFu) : (rb(bd + 4u + s * 8u + within - 8u) >> 4u);
    return float(KV_MXFP4[idx]) * d;
}
#endif

#if defined(FMT_TQ1_0)
// TQ1_0: [u8 qs[48]][u8 qh[4]][f16 d] = 54 bytes, 256 elements. Ternary via pow-3.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 54u;
    float d = f16tof32(ru16(bd + 52u));
    uint src; uint n;
    if (p < 160u) { n = p / 32u; src = rb(bd + (p % 32u)); }
    else if (p < 240u) { uint pp = p - 160u; n = pp / 16u; src = rb(bd + 32u + (pp % 16u)); }
    else { uint pp = p - 240u; n = pp / 4u; src = rb(bd + 48u + (pp % 4u)); }
    uint q = (src * POW3[n]) & 0xFFu;
    uint xi = (q * 3u) >> 8u;
    return float(int(xi) - 1) * d;
}
#endif

#if defined(FMT_TQ2_0)
// TQ2_0: [u8 qs[64]][f16 d] = 66 bytes, 256 elements. 2-bit ternary.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 66u;
    float d = f16tof32(ru16(bd + 64u));
    uint chunk = p / 128u; uint rem = p % 128u; uint l = rem / 32u; uint m = rem % 32u;
    uint q = (rb(bd + chunk * 32u + m) >> (l * 2u)) & 3u;
    return (float(q) - 1.0) * d;
}
#endif

// ── grid-based i-quants (tables from native_grids.glsl) ──
#if defined(FMT_IQ2XXS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 66u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint off = bd + 2u + ib32 * 8u;
    uint aux0 = ru32b(off); uint aux1 = ru32b(off + 4u);
    uint grid_idx = (aux0 >> (8u * l)) & 0xFFu;
    uint sign_idx = (aux1 >> (7u * l)) & 127u;
    float db = d * (0.5 + float(aux1 >> 28u)) * 0.25;
    uint bv = (j < 4u) ? ((G_IQ2XXS[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2XXS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((KSIGNS[sign_idx] >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#endif

#if defined(FMT_IQ2XS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 74u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qs16 = ru16(bd + 2u + (ib32 * 4u + l) * 2u);
    uint grid_idx = qs16 & 511u; uint sign_idx = qs16 >> 9u;
    uint sc = rb(bd + 66u + ib32);
    float dl = (l < 2u) ? (d * (0.5 + float(sc & 0xFu)) * 0.25) : (d * (0.5 + float(sc >> 4u)) * 0.25);
    uint bv = (j < 4u) ? ((G_IQ2XS[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2XS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((KSIGNS[sign_idx] >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return dl * float(sgn8(bv)) * sign;
}
#endif

#if defined(FMT_IQ2S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 82u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qs_byte = rb(bd + 2u + ib32 * 4u + l);
    uint sign_byte = rb(bd + 2u + 32u + ib32 * 4u + l);
    uint qh_byte = rb(bd + 66u + ib32);
    uint grid_idx = qs_byte | ((qh_byte << (8u - 2u * l)) & 0x300u);
    uint sc = rb(bd + 74u + ib32);
    float dl = (l < 2u) ? (d * (0.5 + float(sc & 0xFu)) * 0.25) : (d * (0.5 + float(sc >> 4u)) * 0.25);
    uint bv = (j < 4u) ? ((G_IQ2S[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((sign_byte >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return dl * float(sgn8(bv)) * sign;
}
#endif

#if defined(FMT_IQ3XXS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 98u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j8 = p % 8u;
    uint g1_idx = rb(bd + 2u + ib32 * 8u + 2u * l);
    uint g2_idx = rb(bd + 2u + ib32 * 8u + 2u * l + 1u);
    uint aux32 = ru32b(bd + 66u + 4u * ib32);
    float db = d * (0.5 + float(aux32 >> 28u)) * 0.5;
    uint signs = KSIGNS[(aux32 >> (7u * l)) & 127u];
    uint gidx = (j8 < 4u) ? g1_idx : g2_idx;
    uint bytej = (j8 < 4u) ? j8 : (j8 - 4u);
    uint bv = (G_IQ3XXS[gidx] >> (8u * bytej)) & 0xFFu;
    float sign = (((signs >> j8) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#endif

#if defined(FMT_IQ3S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 110u;
    float d = f16tof32(ru16(bd));
    uint pair = p / 64u; uint grp = (p % 64u) / 32u; uint within = p % 32u; uint l = within / 8u; uint j8 = within % 8u;
    uint sc = rb(bd + 106u + pair);
    float db = (grp == 1u) ? (d * (1.0 + 2.0 * float(sc >> 4u))) : (d * (1.0 + 2.0 * float(sc & 0xFu)));
    uint qh = rb(bd + 66u + pair * 2u + grp);
    uint qs_base = bd + 2u + pair * 16u + grp * 8u;
    uint signs_byte = rb(bd + 74u + pair * 8u + grp * 4u + l);
    uint grididx; uint bytej;
    if (j8 < 4u) { grididx = rb(qs_base + 2u * l) | ((qh << (8u - 2u * l)) & 256u); bytej = j8; }
    else { grididx = rb(qs_base + 2u * l + 1u) | ((qh << (7u - 2u * l)) & 256u); bytej = j8 - 4u; }
    uint bv = (G_IQ3S[grididx] >> (8u * bytej)) & 0xFFu;
    float sign = (((signs_byte >> j8) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#endif

#if defined(FMT_IQ1S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 50u;
    float d = f16tof32(ru16(bd));
    uint ib = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qh = ru16(bd + 34u + 2u * ib);
    float dl = d * (2.0 * float((qh >> 12u) & 7u) + 1.0);
    float delta = ((qh & 0x8000u) != 0u) ? -0.125 : 0.125;
    uint grid_idx = rb(bd + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
    uint bv = (j < 4u) ? ((G_IQ1S[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ1S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    return dl * (float(sgn8(bv)) + delta);
}
#endif

#if defined(FMT_IQ1M)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 56u;
    uint sc0 = ru16(bd + 48u); uint sc1 = ru16(bd + 50u); uint sc2 = ru16(bd + 52u); uint sc3 = ru16(bd + 54u);
    uint d_bits = (sc0 >> 12u) | ((sc1 >> 8u) & 0xF0u) | ((sc2 >> 4u) & 0xF00u) | (sc3 & 0xF000u);
    float d = f16tof32(d_bits);
    uint ib = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint scidx = ib / 2u; uint scw;
    if (scidx == 0u) { scw = sc0; } else if (scidx == 1u) { scw = sc1; }
    else if (scidx == 2u) { scw = sc2; } else { scw = sc3; }
    float dl1 = d * (2.0 * float((scw >> (6u * (ib & 1u))) & 7u) + 1.0);
    float dl2 = d * (2.0 * float((scw >> (6u * (ib & 1u) + 3u)) & 7u) + 1.0);
    float dl = (l >= 2u) ? dl2 : dl1;
    uint qh0 = rb(bd + 32u + ib * 2u); uint qh1 = rb(bd + 32u + ib * 2u + 1u);
    uint grididx; bool deltaneg;
    if (l == 0u) { grididx = rb(bd + ib * 4u) | ((qh0 << 8u) & 0x700u); deltaneg = (qh0 & 0x08u) != 0u; }
    else if (l == 1u) { grididx = rb(bd + ib * 4u + 1u) | ((qh0 << 4u) & 0x700u); deltaneg = (qh0 & 0x80u) != 0u; }
    else if (l == 2u) { grididx = rb(bd + ib * 4u + 2u) | ((qh1 << 8u) & 0x700u); deltaneg = (qh1 & 0x08u) != 0u; }
    else { grididx = rb(bd + ib * 4u + 3u) | ((qh1 << 4u) & 0x700u); deltaneg = (qh1 & 0x80u) != 0u; }
    float delta = deltaneg ? -0.125 : 0.125;
    uint bv = (j < 4u) ? ((G_IQ1S[2u * grididx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ1S[2u * grididx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    return dl * (float(sgn8(bv)) + delta);
}
#endif

// Fallback: formats without an optimized dqblk loop dq() per element (identical math).
#ifndef HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    for (uint w = 0u; w < 32u; w++) { v[w] = dq(gstart + w); }
}
#endif
