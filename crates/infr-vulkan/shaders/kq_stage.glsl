// K-quant weight and int8 activation staging shared by the K-quant prefill GEMMs
// (`native_gemm_i8cm_kq.comp`, int8 cooperative matrix, and `native_gemm_dp4a_kq.comp`, packed
// int8 dot): each k step, `prefetch` reads BK_STEP 32-element blocks of BT activation rows and BW
// weight rows into registers, and `store_to_shared` decodes them into shared memory as packed int8
// words (`As`, `Bs`, row pitch `QP`) with per-block float scales (`Ad`/`Asum` per token, `Bd`/`Bmin`
// per weight row; Q6_K keeps two `Bd` per block, one per 16-element half, and no min).
//
// The includer defines, before including: FMT_Q4K, FMT_Q5K or FMT_Q6K; the constants BT, BW,
// BK_STEP and THREADS (with BT·BK_STEP·2 and BW·BK_STEP·4 multiples of THREADS); the push
// constants `pc` (m, n, k, w_base); the bindings `qa4`, `dact`, `sact`; and includes
// native_weight_addr.glsl with `w_addr` set before the first `prefetch`.
// Shared row pitch in words: BK_STEP blocks of 8 words, padded so neighbouring rows start in
// different banks.
const uint QP = BK_STEP * 8u + 4u;

#ifdef FMT_Q6K
const uint KSCALES = 2u; // one weight scale per 16-element half block
#else
const uint KSCALES = 1u;
#endif

shared uint As[BT * QP];
shared uint Bs[BW * QP];
shared float Ad[BK_STEP * BT];
shared float Bd[BK_STEP * KSCALES * BW];
#ifndef FMT_Q6K
shared float Asum[BK_STEP * BT];
shared float Bmin[BK_STEP * BW];
#endif

// Register staging, one entry per item this thread moves per k step.
const uint A_PER = (BT * BK_STEP * 2u) / THREADS;               // uvec4 halves of a 32 block
const uint AS_PER = (BT * BK_STEP + THREADS - 1u) / THREADS;    // activation scales
const uint B_PER = (BW * BK_STEP * 4u) / THREADS;               // 8-value quarters of a block
const uint BS_PER = (BW * BK_STEP + THREADS - 1u) / THREADS;    // weight scales

uvec4 pre_a[A_PER];
vec2 pre_as[AS_PER];
#if defined(FMT_Q4K)
const uint SB_WORDS = 36u; // 144-byte superblock: d|dmin, scales[12], qs[128]
uvec2 pre_b[B_PER];
uvec4 pre_bs[BS_PER];
#elif defined(FMT_Q5K)
const uint SB_WORDS = 44u; // 176-byte superblock: d|dmin, scales[12], qh[32], qs[128]
uvec4 pre_b[B_PER];        // qs pair, qh pair
uvec4 pre_bs[BS_PER];
#else
const uint SB_BYTES = 210u; // ql[128], qh[64], scales[16] int8, d f16 — only 2-byte aligned
uvec3 pre_bl[B_PER];
uvec3 pre_bh[B_PER];
uvec2 pre_bs[BS_PER];
#endif

// The 32-element weight block a (row, step-local block) item reads.
uint wblock(uint o0, uint row, uint kb, uint ks) {
    return (pc.w_base + (o0 + row) * pc.k) / 32u + kb + ks;
}

#ifdef FMT_Q6K
// Eight bytes at a 2-byte-aligned offset: three words read raw; `join8` shifts them into place
// once the loads have landed, so the shift costs nothing while the reads are in flight.
uvec3 read8(uint byte_off) {
    uint wi = byte_off >> 2u;
    return uvec3(arena_word(wi), arena_word(wi + 1u), arena_word(wi + 2u));
}
uvec2 join8(uvec3 w, uint byte_off) {
    if ((byte_off & 2u) == 0u) {
        return w.xy;
    }
    return uvec2((w.x >> 16u) | (w.y << 16u), (w.y >> 16u) | (w.z << 16u));
}
uint half16(uint word, uint byte_off) {
    return (byte_off & 2u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
}
// Signed 6-bit value q − 32 in each byte of a packed word.
uint recenter32(uint v) {
    return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}
#endif

void prefetch(uint t0, uint o0, uint kb, uint tid) {
    uint nblk = pc.k / 32u;
    [[unroll]] for (uint i = 0u; i < A_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / (BK_STEP * 2u);
        uint q = idx % (BK_STEP * 2u);
        uint row = t0 + r;
        pre_a[i] = row < pc.m ? qa4[row * (pc.k / 16u) + kb * 2u + q] : uvec4(0u);
    }
    [[unroll]] for (uint i = 0u; i < AS_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / BK_STEP;
        uint ks = idx % BK_STEP;
        uint row = t0 + r;
        bool live = idx < BT * BK_STEP && row < pc.m;
        uint at = row * nblk + kb + ks;
        pre_as[i] = live ? vec2(float(dact[at]), float(sact[at])) : vec2(0.0);
    }
    [[unroll]] for (uint i = 0u; i < B_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / (BK_STEP * 4u);
        uint ks = (idx / 4u) % BK_STEP;
        uint lr = idx % 4u;
        uint gb = wblock(o0, r, kb, ks);
        uint sb = gb >> 3u;
        uint sub = gb & 7u;
#if defined(FMT_Q4K)
        pre_b[i] = arena_word2(sb * SB_WORDS + 4u + (sub >> 1u) * 8u + lr * 2u);
#elif defined(FMT_Q5K)
        uvec2 qs = arena_word2(sb * SB_WORDS + 12u + (sub >> 1u) * 8u + lr * 2u);
        uvec2 qh = arena_word2(sb * SB_WORDS + 4u + lr * 2u);
        pre_b[i] = uvec4(qs, qh);
#else
        uint g = sub >> 2u;
        uint j = sub & 3u;
        uint base = sb * SB_BYTES;
        pre_bl[i] = read8(base + g * 64u + (j & 1u) * 32u + lr * 8u);
        pre_bh[i] = read8(base + 128u + g * 32u + lr * 8u);
#endif
    }
    [[unroll]] for (uint i = 0u; i < BS_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / BK_STEP;
        uint ks = idx % BK_STEP;
        if (idx < BW * BK_STEP) {
            uint gb = wblock(o0, r, kb, ks);
            uint sb = gb >> 3u;
#if defined(FMT_Q6K)
            uint base = sb * SB_BYTES;
            uint sub = gb & 7u;
            pre_bs[i] = uvec2(arena_word((base + 192u + sub * 2u) >> 2u),
                              arena_word((base + 208u) >> 2u));
#else
            pre_bs[i] = arena_word4(sb * SB_WORDS);
#endif
        }
    }
}

void store_to_shared(uint o0, uint kb, uint tid) {
    [[unroll]] for (uint i = 0u; i < A_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / (BK_STEP * 2u);
        uint q = idx % (BK_STEP * 2u);
        uint at = r * QP + q * 4u;
        As[at] = pre_a[i].x;
        As[at + 1u] = pre_a[i].y;
        As[at + 2u] = pre_a[i].z;
        As[at + 3u] = pre_a[i].w;
    }
    [[unroll]] for (uint i = 0u; i < AS_PER; i++) {
        uint idx = tid + i * THREADS;
        if (idx < BT * BK_STEP) {
            uint r = idx / BK_STEP;
            uint ks = idx % BK_STEP;
            Ad[ks * BT + r] = pre_as[i].x;
#ifndef FMT_Q6K
            Asum[ks * BT + r] = pre_as[i].y;
#endif
        }
    }
    [[unroll]] for (uint i = 0u; i < B_PER; i++) {
        uint idx = tid + i * THREADS;
        uint r = idx / (BK_STEP * 4u);
        uint ks = (idx / 4u) % BK_STEP;
        uint lr = idx % 4u;
        uint sub = wblock(o0, r, kb, ks) & 7u;
        uvec2 v;
#if defined(FMT_Q4K)
        v = (pre_b[i] >> ((sub & 1u) * 4u)) & 0x0F0F0F0Fu;
#elif defined(FMT_Q5K)
        v = (pre_b[i].xy >> ((sub & 1u) * 4u)) & 0x0F0F0F0Fu;
        v |= ((pre_b[i].zw >> sub) & 0x01010101u) << 4u;
#else
        uint gb = wblock(o0, r, kb, ks);
        uint base = (gb >> 3u) * SB_BYTES;
        uint g = sub >> 2u;
        uint j = sub & 3u;
        uvec2 ql = join8(pre_bl[i], base + g * 64u + (j & 1u) * 32u + lr * 8u);
        uvec2 qh = join8(pre_bh[i], base + 128u + g * 32u + lr * 8u);
        ql = (ql >> ((j >> 1u) * 4u)) & 0x0F0F0F0Fu;
        v = ql | (((qh >> (j * 2u)) & 0x03030303u) << 4u);
        v = uvec2(recenter32(v.x), recenter32(v.y));
#endif
        uint at = r * QP + ks * 8u + lr * 2u;
        Bs[at] = v.x;
        Bs[at + 1u] = v.y;
    }
    [[unroll]] for (uint i = 0u; i < BS_PER; i++) {
        uint idx = tid + i * THREADS;
        if (idx < BW * BK_STEP) {
            uint r = idx / BK_STEP;
            uint ks = idx % BK_STEP;
            uint gb = wblock(o0, r, kb, ks);
            uint sub = gb & 7u;
#if defined(FMT_Q6K)
            uint base = (gb >> 3u) * SB_BYTES;
            uint sc = half16(pre_bs[i].x, base + 192u + sub * 2u);
            float d = unpackHalf2x16(half16(pre_bs[i].y, base + 208u)).x;
            Bd[(ks * KSCALES) * BW + r] = d * float(int(sc << 24u) >> 24);
            Bd[(ks * KSCALES + 1u) * BW + r] = d * float(int(sc << 16u) >> 24);
#else
            // get_scale_min_k4: for j < 4 the 6-bit scale/min sit in bytes j and j+4; above
            // that they are split across byte j+4 (low bits) and the top two bits of bytes
            // j-4 and j. Byte b of the scales lives in word 1 + b/4 at lane b%4.
            uint j = sub & 3u;
            uint s_j = (pre_bs[i].y >> (8u * j)) & 0xFFu;
            uint s_j4 = (pre_bs[i].z >> (8u * j)) & 0xFFu;
            uint s_j8 = (pre_bs[i].w >> (8u * j)) & 0xFFu;
            uint sc = sub < 4u ? (s_j & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j >> 6u) << 4u));
            uint mn = sub < 4u ? (s_j4 & 0x3Fu) : ((s_j8 >> 4u) | ((s_j4 >> 6u) << 4u));
            vec2 dm = unpackHalf2x16(pre_bs[i].x);
            Bd[ks * BW + r] = dm.x * float(sc);
            Bmin[ks * BW + r] = -dm.y * float(mn);
#endif
        }
    }
}
