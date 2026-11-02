//! Native EXL3 projection geometry and logical input/output ordering.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Exl3Codebook {
    Cb0 = 0,
    Mcg = 1,
    Mul1 = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exl3Layout {
    #[default]
    Identity,
    RopeInterleaved {
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    },
    GroupedValueHeads {
        prefix: u32,
        key_heads: u32,
        value_heads: u32,
        head_dim: u32,
    },
}

pub type Exl3OutputLayout = Exl3Layout;

impl Exl3Layout {
    pub fn validate(self, width: u32) -> Result<()> {
        let valid = match self {
            Self::Identity => true,
            Self::RopeInterleaved {
                heads,
                head_dim,
                rotary_dim,
            } => {
                heads > 0
                    && head_dim > 0
                    && heads.checked_mul(head_dim) == Some(width)
                    && rotary_dim <= head_dim
                    && rotary_dim.is_multiple_of(2)
            }
            Self::GroupedValueHeads {
                prefix,
                key_heads,
                value_heads,
                head_dim,
            } => {
                key_heads > 0
                    && value_heads > 0
                    && head_dim > 0
                    && value_heads.is_multiple_of(key_heads)
                    && value_heads
                        .checked_mul(head_dim)
                        .and_then(|n| prefix.checked_add(n))
                        == Some(width)
            }
        };
        if valid && width > 0 {
            Ok(())
        } else {
            Err(Error::Backend("exl3: invalid element layout".into()))
        }
    }

    /// Map a physical element index after validating the layout dimensions.
    pub fn index(self, natural: usize) -> usize {
        match self {
            Self::Identity => natural,
            Self::RopeInterleaved {
                head_dim,
                rotary_dim,
                ..
            } => {
                let hd = head_dim as usize;
                let rd = rotary_dim as usize;
                let j = natural % hd;
                natural - j
                    + if j < rd / 2 {
                        2 * j
                    } else if j < rd {
                        2 * (j - rd / 2) + 1
                    } else {
                        j
                    }
            }
            Self::GroupedValueHeads {
                prefix,
                key_heads,
                value_heads,
                head_dim,
            } => {
                let prefix = prefix as usize;
                if natural < prefix {
                    return natural;
                }
                let hd = head_dim as usize;
                let keys = key_heads as usize;
                let per_key = value_heads as usize / keys;
                let head = (natural - prefix) / hd;
                prefix + (keys * (head % per_key) + head / per_key) * hd + (natural - prefix) % hd
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3EmbeddingSpec {
    pub width: u32,
    pub vocab: u32,
    pub bits: u32,
}

impl Exl3EmbeddingSpec {
    pub fn row_bytes(self) -> Result<usize> {
        if self.width == 0 || self.vocab == 0 || !(1..=8).contains(&self.bits) {
            return Err(Error::Backend(
                "exl3 embedding: invalid dimensions or rate".into(),
            ));
        }
        (self.width.div_ceil(256) as usize)
            .checked_mul(2 * (16 * self.bits as usize + 1))
            .ok_or_else(|| Error::Backend("exl3 embedding: row size overflow".into()))
    }
    pub fn trellis_bytes(self) -> Result<usize> {
        self.row_bytes()?
            .checked_mul(self.vocab as usize)
            .ok_or_else(|| Error::Backend("exl3 embedding: table size overflow".into()))
    }
    pub fn sign_bytes(self) -> Result<usize> {
        self.row_bytes()?;
        (self.width.div_ceil(256) as usize)
            .checked_mul(512)
            .ok_or_else(|| Error::Backend("exl3 embedding: sign size overflow".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3Spec {
    pub in_f: u32,
    pub out_f: u32,
    pub bits: u32,
    pub codebook: Exl3Codebook,
    pub input_layout: Exl3Layout,
    pub output_layout: Exl3Layout,
}

impl Exl3Spec {
    pub fn trellis_bytes(self) -> Result<usize> {
        if !(1..=8).contains(&self.bits)
            || self.in_f == 0
            || self.out_f == 0
            || !self.in_f.is_multiple_of(128)
            || !self.out_f.is_multiple_of(128)
        {
            return Err(Error::Backend(
                "exl3: invalid dimensions or integer rate".into(),
            ));
        }
        self.input_layout.validate(self.in_f)?;
        self.output_layout.validate(self.out_f)?;
        (self.in_f as usize)
            .checked_mul(self.out_f as usize)
            .and_then(|n| n.checked_mul(self.bits as usize))
            .map(|n| n / 8)
            .ok_or_else(|| Error::Backend("exl3: trellis size overflow".into()))
    }

    pub fn input_index(self, natural: usize) -> usize {
        self.input_layout.index(natural)
    }
    pub fn output_index(self, natural: usize) -> usize {
        self.output_layout.index(natural)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_scatter_is_permutation_with_partial_rotary_heads() {
        let spec = Exl3Spec {
            in_f: 128,
            out_f: 256,
            bits: 4,
            codebook: Exl3Codebook::Cb0,
            input_layout: Exl3Layout::Identity,
            output_layout: Exl3OutputLayout::RopeInterleaved {
                heads: 4,
                head_dim: 64,
                rotary_dim: 48,
            },
        };
        assert_eq!(spec.trellis_bytes().unwrap(), 16384);
        let mut seen = [false; 256];
        for i in 0..256 {
            let j = spec.output_index(i);
            assert!(!seen[j]);
            seen[j] = true;
            if i % 64 >= 48 {
                assert_eq!(i, j);
            }
        }
        assert_eq!(spec.output_index(24), 1);
        assert_eq!(spec.output_index(64), 64);
    }

    #[test]
    fn grouped_value_heads_map_physical_to_tiled_with_prefix() {
        let layout = Exl3Layout::GroupedValueHeads {
            prefix: 128,
            key_heads: 3,
            value_heads: 6,
            head_dim: 64,
        };
        layout.validate(512).unwrap();
        let mut seen = [false; 512];
        for i in 0..512 {
            let j = layout.index(i);
            assert!(!seen[j]);
            seen[j] = true;
            if i < 128 {
                assert_eq!(j, i);
            }
        }
        for (physical, tiled) in [0, 3, 1, 4, 2, 5].into_iter().enumerate() {
            assert_eq!(layout.index(128 + physical * 64 + 7), 128 + tiled * 64 + 7);
        }
        let spec = Exl3Spec {
            in_f: 4096,
            out_f: 8192,
            bits: 2,
            codebook: Exl3Codebook::Mcg,
            input_layout: Exl3Layout::GroupedValueHeads {
                prefix: 0,
                key_heads: 16,
                value_heads: 32,
                head_dim: 128,
            },
            output_layout: Exl3Layout::GroupedValueHeads {
                prefix: 4096,
                key_heads: 16,
                value_heads: 32,
                head_dim: 128,
            },
        };
        spec.trellis_bytes().unwrap();
        assert_eq!(spec.input_index(128), 2048);
        assert_eq!(spec.output_index(4096 + 128), 6144);
    }

    #[test]
    fn rejects_invalid_head_layouts() {
        for layout in [
            Exl3Layout::GroupedValueHeads {
                prefix: 0,
                key_heads: 0,
                value_heads: 32,
                head_dim: 128,
            },
            Exl3Layout::GroupedValueHeads {
                prefix: 0,
                key_heads: 3,
                value_heads: 32,
                head_dim: 128,
            },
            Exl3Layout::GroupedValueHeads {
                prefix: 1,
                key_heads: 16,
                value_heads: 32,
                head_dim: 128,
            },
            Exl3Layout::GroupedValueHeads {
                prefix: 0,
                key_heads: 16,
                value_heads: 32,
                head_dim: 0,
            },
            Exl3Layout::GroupedValueHeads {
                prefix: u32::MAX,
                key_heads: 1,
                value_heads: 32,
                head_dim: 128,
            },
        ] {
            assert!(layout.validate(4096).is_err());
        }
    }
}
