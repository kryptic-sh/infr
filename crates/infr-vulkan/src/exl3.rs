//! Portable native EXL3 shader binaries.

use std::sync::OnceLock;

macro_rules! shader {
    ($func:ident, $name:literal) => {
        pub(crate) fn $func() -> &'static [u32] {
            static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
            WORDS.get_or_init(|| {
                crate::gemm::spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
        }
    };
}

shader!(hadamard_in, "exl3_hadamard_in");
shader!(hadamard_out, "exl3_hadamard_out");
shader!(gemv, "exl3_gemv");
shader!(gemv_f16, "exl3_gemv_f16");
shader!(prefill, "exl3_prefill");
