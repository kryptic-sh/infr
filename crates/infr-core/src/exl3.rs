//! Native EXL3 projection geometry and output ordering.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3Codebook {
    Cb0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3OutputLayout {
    Identity,
    RopeInterleaved {
        heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3Spec {
    pub in_f: u32,
    pub out_f: u32,
    pub bits: u32,
    pub codebook: Exl3Codebook,
    pub output_layout: Exl3OutputLayout,
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
        if let Exl3OutputLayout::RopeInterleaved {
            heads,
            head_dim,
            rotary_dim,
        } = self.output_layout
        {
            if heads == 0
                || head_dim == 0
                || heads.checked_mul(head_dim) != Some(self.out_f)
                || rotary_dim > head_dim
                || !rotary_dim.is_multiple_of(2)
            {
                return Err(Error::Backend("exl3: invalid rotary output layout".into()));
            }
        }
        (self.in_f as usize)
            .checked_mul(self.out_f as usize)
            .and_then(|n| n.checked_mul(self.bits as usize))
            .map(|n| n / 8)
            .ok_or_else(|| Error::Backend("exl3: trellis size overflow".into()))
    }

    pub fn output_index(self, natural: usize) -> usize {
        match self.output_layout {
            Exl3OutputLayout::Identity => natural,
            Exl3OutputLayout::RopeInterleaved {
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
        }
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
}
