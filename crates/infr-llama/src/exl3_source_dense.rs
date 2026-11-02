use super::*;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum DenseOp {
    #[default]
    Identity,
    AddOne,
    NegativeExp,
}

#[derive(Clone, Copy, Default)]
pub(super) struct DenseOptions {
    pub f32: bool,
    pub op: DenseOp,
    pub layout: Exl3Layout,
}

impl Exl3Source {
    pub(super) fn add_dense_with(
        &mut self,
        name: &str,
        hf: &str,
        shape: &[usize],
        source_shape: Option<&[usize]>,
        options: DenseOptions,
    ) -> Result<()> {
        let info = find_hf(&self.st, hf)?;
        let expected: Vec<_> = source_shape
            .map(<[usize]>::to_vec)
            .unwrap_or_else(|| shape.iter().rev().copied().collect());
        let valid_dtype = if options.f32 {
            matches!(info.dtype, DType::F16 | DType::Bf16 | DType::F32)
        } else {
            info.dtype == DType::F16
        };
        ensure!(
            valid_dtype && info.shape == expected,
            "exl3: {hf} has invalid dense dtype or shape: {:?} {:?}, expected {expected:?}",
            info.dtype,
            info.shape
        );
        ensure!(
            dense_size(shape, 1)? == dense_size(&expected, 1)?,
            "exl3: {hf} reshape changes element count"
        );
        let count = dense_size(shape, 1)?;
        options.layout.validate(u32::try_from(count)?)?;
        let dtype = if options.f32 || shape.len() == 1 || options.op != DenseOp::Identity {
            DType::F32
        } else {
            DType::F16
        };
        let nbytes = dense_size(
            shape,
            dtype
                .dense_bytes(1)
                .context("exl3: invalid dense output dtype")?,
        )?;
        self.infos.push(TensorInfo {
            name: name.into(),
            shape: shape.into(),
            dtype,
            offset: 0,
            nbytes,
        });
        self.passthrough.insert(
            name.into(),
            DenseSpec {
                hf_name: hf.into(),
                dtype,
                convert: options.f32
                    || dtype != info.dtype
                    || options.op != DenseOp::Identity
                    || options.layout != Exl3Layout::Identity,
                op: options.op,
                layout: options.layout,
                cache: OnceLock::new(),
            },
        );
        Ok(())
    }

    pub(super) fn dense_bytes<'a>(&'a self, spec: &'a DenseSpec) -> Result<&'a [u8]> {
        if !spec.convert {
            return Ok(self.st.tensor_bytes(&spec.hf_name)?);
        }
        match spec
            .cache
            .get_or_init(|| self.transform_dense(spec).map_err(|e| e.to_string()))
        {
            Ok(bytes) => Ok(bytes),
            Err(error) => Err(anyhow!("{error}")),
        }
    }

    fn transform_dense(&self, spec: &DenseSpec) -> Result<Vec<u8>> {
        let info = find_hf(&self.st, &spec.hf_name)?;
        let raw = self.st.tensor_bytes(&spec.hf_name)?;
        let source_width = info
            .dtype
            .dense_bytes(1)
            .context("exl3: invalid dense source dtype")?;
        let width = spec
            .dtype
            .dense_bytes(1)
            .context("exl3: invalid dense output dtype")?;
        let count = raw.len() / source_width;
        let mut out = vec![
            0;
            count
                .checked_mul(width)
                .context("exl3: dense conversion size overflow")?
        ];
        for (i, bytes) in raw.chunks_exact(source_width).enumerate() {
            let value = match info.dtype {
                DType::F16 => half::f16::from_le_bytes(bytes.try_into()?).to_f32(),
                DType::Bf16 => {
                    f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into()?)) << 16)
                }
                DType::F32 => f32::from_le_bytes(bytes.try_into()?),
                _ => return Err(anyhow!("exl3: unsupported dense dtype")),
            };
            ensure!(
                value.is_finite(),
                "exl3: {} contains non-finite values",
                spec.hf_name
            );
            let value = match spec.op {
                DenseOp::Identity => value,
                DenseOp::AddOne => value + 1.0,
                DenseOp::NegativeExp => -value.exp(),
            };
            ensure!(
                value.is_finite(),
                "exl3: {} conversion overflows",
                spec.hf_name
            );
            let j = spec.layout.index(i) * width;
            if spec.dtype == DType::F32 {
                out[j..j + 4].copy_from_slice(&value.to_le_bytes());
            } else {
                out[j..j + 2].copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
            }
        }
        Ok(out)
    }
}
