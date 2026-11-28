// RoPE angle arithmetic shared by every rotary shader (rope, qk_norm_rope,
// qk_norm_rope_interleaved, mla), so the GPU angle is the CPU reference's
// (infr_core::math::rope: `pos * rope_freqs[p]`, then `/ divisor`, each rounded to binary32) and
// its sine and cosine are accurate. The frequencies themselves are the CPU's `rope_freqs` table,
// built on the host and bound to each shader, so only the product, the divide and the sincos
// happen here. The GLSL built-ins are not good enough for those: Vulkan lets `/` be off by 2.5
// ulp and `sin`/`cos` by an absolute 2^-11, and drivers use that slack (Mesa ANV on Intel does).
//
// Exactness rests only on binary32 add, subtract and multiply, which Vulkan requires to be
// correctly rounded, with `precise` so no compiler reassociates or contracts them, and on integer
// operations. Not `fma`: Vulkan lets it round like a multiply then an add, and Mesa folds
// `fma(a, b, -(a * b))` to 0 even under `precise`. Exact products therefore come from Dekker's
// splitting, not from an fma. The GPU test `rope_angle_matches_f64` holds every function here to
// the CPU or to f64.

// Correctly rounded `a / b` (ties to even) for a normal or zero `a` and a normal `b`: restoring
// long division of the significands. A quotient past the normal range flushes to signed zero or
// infinity; no RoPE angle comes near either (a "never rotate" divisor of 1e30 leaves it normal).
float rope_div(float a, float b) {
    uint ua = floatBitsToUint(a);
    uint ub = floatBitsToUint(b);
    uint sign = (ua ^ ub) & 0x80000000u;
    if ((ua & 0x7fffffffu) == 0u) {
        return uintBitsToFloat(sign);
    }
    uint ma = (ua & 0x007fffffu) | 0x00800000u;
    uint mb = (ub & 0x007fffffu) | 0x00800000u;
    int ex = int((ua >> 23) & 0xffu) - int((ub >> 23) & 0xffu) + 127;
    if (ma < mb) {
        ma <<= 1;
        ex -= 1;
    }
    // ma / mb is in [1, 2): 24 quotient bits, then the round bit, then the rest as sticky.
    uint q = 0u;
    uint r = ma;
    for (int i = 0; i < 25; i++) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    uint m = q >> 1;
    if ((q & 1u) != 0u && (r != 0u || (m & 1u) != 0u)) {
        m += 1u;
    }
    if (m == 0x01000000u) {
        m >>= 1;
        ex += 1;
    }
    if (ex <= 0) {
        return uintBitsToFloat(sign);
    }
    if (ex >= 255) {
        return uintBitsToFloat(sign | 0x7f800000u);
    }
    return uintBitsToFloat(sign | (uint(ex) << 23) | (m & 0x007fffffu));
}

// The exact product a * b as hi + lo: Veltkamp's split into 12-bit halves (by 2^12 + 1), then
// Dekker's two-product.
vec2 rope_two_prod(float a, float b) {
    precise float ca = 4097.0 * a;
    precise float ah = ca - (ca - a);
    precise float al = a - ah;
    precise float cb = 4097.0 * b;
    precise float bh = cb - (cb - b);
    precise float bl = b - bh;
    precise float p = a * b;
    precise float e = (((ah * bh - p) + ah * bl) + al * bh) + al * bl;
    return vec2(p, e);
}

// x - k*pi/2 for an integer-valued k below 2^24, pi/2 split in three binary32 parts (Cody-Waite).
// k * PIO2_1 is exact as a two-product and x minus its high half is exact (Sterbenz); each later
// subtraction rounds once at the size of its operands, at most |r| + |k| * 2^-24 here. So the
// error is absolute, not relative to r: up to about 2^-24 as |x| nears 2^24, whatever r's size.
// The dropped tail of pi/2 adds |k| * 2^-76.
float rope_pio2_rem(float x, float k) {
    const float PIO2_1 = uintBitsToFloat(0x3fc90fdbu);
    const float PIO2_2 = uintBitsToFloat(0xb33bbd2eu);
    const float PIO2_3 = uintBitsToFloat(0xa6f72cedu);
    vec2 p1 = rope_two_prod(k, PIO2_1);
    vec2 p2 = rope_two_prod(k, PIO2_2);
    precise float r = (((x - p1.x) - p1.y) - p2.x) - (p2.y + k * PIO2_3);
    return r;
}

// ROPE_SINCOS_MAX, the largest |x| rope_sincos takes: infr_core::math::ROPE_ANGLE_MAX, written
// here by build.rs. Past it a binary32 position is no longer an exact integer, so the CPU
// reference's angle stops meaning a position too; the seam refuses a context that would reach it
// (`refuse_rope_past_angle_limit`). Should an angle pass it anyway the result is NaN, which is not
// a reliable signal (NaN need not propagate without shaderSignedZeroInfNanPreserve) but is never
// a plausible rotation.
#include "rope_limits.glsl"

// (sin x, cos x) for binary32 |x| <= ROPE_SINCOS_MAX.
//
// k = round(x * 2/pi), r = x - k*pi/2 (rope_pio2_rem). The product is rounded to binary32, so
// near an odd multiple of pi/4 k can be one off once |x| is large enough for the product's
// error to approach 1/2; |r| then exceeds pi/4, and stepping k toward r and reducing again
// brings it back (over the domain the product's error stays below 1, so one step is enough).
// Polynomials: Cephes sinf/cosf (S. Moshier), minimax on |r| <= pi/4, about 1 ulp each.
vec2 rope_sincos(float x) {
    if (!(abs(x) <= ROPE_SINCOS_MAX)) {
        return vec2(uintBitsToFloat(0x7fc00000u));
    }
    const float TWO_OVER_PI = uintBitsToFloat(0x3f22f983u);
    const float PIO4 = uintBitsToFloat(0x3f490fdbu);
    precise float k = roundEven(x * TWO_OVER_PI);
    float r = rope_pio2_rem(x, k);
    if (abs(r) > PIO4) {
        k += sign(r);
        r = rope_pio2_rem(x, k);
    }
    precise float z = r * r;
    precise float s = r + r * z * ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1);
    precise float c = (1.0 - 0.5 * z)
        + z * z * ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2);
    // Quadrant k mod 4 (two's complement keeps it right for negative k).
    uint q = uint(int(k)) & 3u;
    vec2 sc = (q & 1u) == 0u ? vec2(s, c) : vec2(c, -s);
    return (q & 2u) == 0u ? sc : -sc;
}
