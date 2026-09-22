//! Declaring round-to-nearest-even for 16-bit floats on a compiled SPIR-V module.
//!
//! SPIR-V leaves the rounding of a conversion to `float16_t` implementation-defined unless the
//! module declares a rounding mode, and drivers disagree: RADV rounds to nearest even, while AMD's
//! proprietary driver truncates on RDNA2 — so every f16 the shaders store (the KV cache, the Q8
//! activation scales) came out up to one ulp low there, and a device's arithmetic stopped matching
//! every other device's. `glslc` has no flag for execution modes, so rather than editing every
//! shader and building each one twice (llama.cpp's `_rte` variants), [`with_rte_f16`] adds the
//! declaration to a module at pipeline-creation time. The caller decides where: only on devices
//! that report `shaderRoundingModeRTEFloat16` (the declaration is invalid elsewhere) AND were seen
//! to truncate by default (see `VulkanBackend::arm_f16_rounding`).
//!
//! Transcribed from the SPIR-V specification (unified, 1.6): the physical layout (§2.3), the
//! logical layout (§2.4) and the enumerant values below.

/// SPIR-V's magic number, the module's first word.
const MAGIC: u32 = 0x0723_0203;
/// Words before the first instruction: magic, version, generator, bound, schema.
const HEADER_WORDS: usize = 5;
/// Float controls are core from SPIR-V 1.4; earlier modules would also need
/// `OpExtension "SPV_KHR_float_controls"`, which is not worth supporting — every shader here is
/// built for `vulkan1.3`, i.e. SPIR-V 1.6.
const MIN_VERSION: u32 = 0x0001_0400;

const OP_CAPABILITY: u32 = 17;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_FLOAT: u32 = 22;
/// `Capability RoundingModeRTE`.
const CAPABILITY_ROUNDING_MODE_RTE: u32 = 4467;
/// `ExecutionMode RoundingModeRTE`, whose one literal is the float width it applies to.
const EXECUTION_MODE_ROUNDING_MODE_RTE: u32 = 4462;
const F16_WIDTH: u32 = 16;

/// One instruction's first word: word count in the high half, opcode in the low.
const fn inst_word(word_count: u32, opcode: u32) -> u32 {
    (word_count << 16) | opcode
}

/// `spv` with `RoundingModeRTE` for 16-bit floats declared on every entry point, or `None` when
/// there is nothing to do: the module has no 16-bit float type, already declares the capability,
/// predates SPIR-V 1.4, or is not well-formed enough to patch safely. `None` means "use `spv`
/// unchanged", so a malformed module still reaches the driver's own validation, not a panic here.
pub(crate) fn with_rte_f16(spv: &[u32]) -> Option<Vec<u32>> {
    if spv.len() < HEADER_WORDS || spv[0] != MAGIC || spv[1] < MIN_VERSION {
        return None;
    }
    let mut last_capability_end = None;
    let mut last_entry_point_end = None;
    let mut entry_points = Vec::new();
    let mut has_f16 = false;
    let mut at = HEADER_WORDS;
    while at < spv.len() {
        let word_count = (spv[at] >> 16) as usize;
        let opcode = spv[at] & 0xffff;
        if word_count == 0 || at + word_count > spv.len() {
            return None;
        }
        let operands = &spv[at + 1..at + word_count];
        match opcode {
            OP_CAPABILITY => {
                if operands.first() == Some(&CAPABILITY_ROUNDING_MODE_RTE) {
                    return None;
                }
                last_capability_end = Some(at + word_count);
            }
            // Operands: execution model, entry-point <id>, name, interface <id>s.
            OP_ENTRY_POINT => {
                entry_points.push(*operands.get(1)?);
                last_entry_point_end = Some(at + word_count);
            }
            // Operands: result <id>, width (then an optional encoding).
            OP_TYPE_FLOAT => has_f16 |= operands.get(1) == Some(&F16_WIDTH),
            _ => {}
        }
        at += word_count;
    }
    let (capability_at, modes_at) = (last_capability_end?, last_entry_point_end?);
    if !has_f16 || modes_at < capability_at {
        return None;
    }

    let mut out = Vec::with_capacity(spv.len() + 2 + 4 * entry_points.len());
    out.extend_from_slice(&spv[..capability_at]);
    out.extend([inst_word(2, OP_CAPABILITY), CAPABILITY_ROUNDING_MODE_RTE]);
    out.extend_from_slice(&spv[capability_at..modes_at]);
    // Execution modes come straight after the entry points (logical layout, §2.4), in any order.
    for id in entry_points {
        out.extend([
            inst_word(4, OP_EXECUTION_MODE),
            id,
            EXECUTION_MODE_ROUNDING_MODE_RTE,
            F16_WIDTH,
        ]);
    }
    out.extend_from_slice(&spv[modes_at..]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal compute module: `Shader` and `Float16` capabilities, a memory model, one entry
    /// point `%1 "main"`, its `LocalSize`, and `%2 = OpTypeFloat width`. Only the layout matters;
    /// no body is needed to exercise the patch.
    fn module(version: u32, float_width: u32) -> Vec<u32> {
        let name = u32::from_le_bytes(*b"main");
        vec![
            MAGIC,
            version,
            0,
            3,
            0,
            inst_word(2, OP_CAPABILITY),
            1, // Shader
            inst_word(2, OP_CAPABILITY),
            9, // Float16
            inst_word(3, 14),
            0,
            1, // OpMemoryModel Logical GLSL450
            inst_word(5, OP_ENTRY_POINT),
            5, // GLCompute
            1,
            name,
            0, // "main\0" padded to a whole word
            inst_word(6, OP_EXECUTION_MODE),
            1,
            17, // LocalSize 64 1 1
            64,
            1,
            1,
            inst_word(3, OP_TYPE_FLOAT),
            2,
            float_width,
        ]
    }

    #[test]
    fn declares_the_capability_and_the_mode_where_the_layout_puts_them() {
        let spv = module(0x0001_0600, 16);
        let got = with_rte_f16(&spv).expect("an f16 module is patched");
        let mut want = spv[..9].to_vec();
        want.extend([inst_word(2, OP_CAPABILITY), CAPABILITY_ROUNDING_MODE_RTE]);
        want.extend_from_slice(&spv[9..17]);
        want.extend([inst_word(4, OP_EXECUTION_MODE), 1, 4462, 16]);
        want.extend_from_slice(&spv[17..]);
        assert_eq!(got, want);
        assert_eq!(with_rte_f16(&got), None, "patching twice must be a no-op");
    }

    #[test]
    fn leaves_modules_it_cannot_or_need_not_patch_alone() {
        assert_eq!(with_rte_f16(&module(0x0001_0600, 32)), None, "no f16 type");
        assert_eq!(with_rte_f16(&module(0x0001_0300, 16)), None, "SPIR-V 1.3");
        let mut truncated = module(0x0001_0600, 16);
        truncated.truncate(truncated.len() - 1);
        assert_eq!(
            with_rte_f16(&truncated),
            None,
            "instruction runs past the end"
        );
        let mut not_spirv = module(0x0001_0600, 16);
        not_spirv[0] = 0;
        assert_eq!(with_rte_f16(&not_spirv), None, "bad magic");
    }

    /// A real shader from this crate's build, not just the hand-written module above: the patch
    /// finds its f16 type and entry point, and adds exactly the two instructions (6 words).
    #[test]
    fn patches_a_real_f16_shader() {
        let spv = crate::gemm::store_f16_spv();
        let got = with_rte_f16(spv).expect("store_f16 (the KV-cache store) has an f16 type");
        assert_eq!(got.len(), spv.len() + 6);
    }
}
