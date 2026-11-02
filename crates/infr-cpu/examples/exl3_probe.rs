//! Probe a real EXL3 checkpoint's trellis tensors through the CPU oracle.
//!
//! Reads a `.safetensors` file, finds every `*.trellis` tensor, detects its
//! codebook from the sibling `.mcg` / `.mul1` marker tensors (absent = cb0
//! `3inst`, per `quantize.py::codebook_markers`), decodes a sample of its tiles
//! with [`infr_cpu::exl3`], and reports distribution stats. This is the Phase 0
//! "parse trellis bytes -> f32" step exercised on REAL quantized bytes (not
//! just PRBS tiles): every sampled tile must decode 100% finite with a sane
//! spread, or the port misreads the checkpoint layout.
//!
//! Layout reference: `tests/testlib/exl3.py` (`trellis (k//16, n//16, 16*K)`
//! int16) and `modules/quant/exl3.py` (codebook flags from marker presence) in
//! turboderp-org/exllamav3 @ `bd1f519172406a9560bd5c90e61000691bb69a07`.
//!
//! Run: `cargo run -p infr-cpu --example exl3_probe -- <model.safetensors> [max_tiles]`

use infr_cpu::exl3::{decode_tile, Codebook};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: exl3_probe <model.safetensors> [max_tiles_per_tensor]");
    let max_tiles: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("max_tiles must be a number"))
        .unwrap_or(4);

    let mut f = File::open(&path)?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let hlen = u64::from_le_bytes(len8) as usize;
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf)?;
    let header: serde_json::Value = serde_json::from_slice(&hbuf)?;
    let data_start = 8 + hlen;

    let obj = header.as_object().expect("header must be a JSON object");
    // Codebook markers: `<prefix>.mcg` / `<prefix>.mul1` next to `<prefix>.trellis`.
    let mut markers: HashMap<String, Codebook> = HashMap::new();
    for k in obj.keys() {
        if let Some(p) = k.strip_suffix(".mcg") {
            markers.insert(p.to_string(), Codebook::Cb1);
        } else if let Some(p) = k.strip_suffix(".mul1") {
            markers.insert(p.to_string(), Codebook::Cb2);
        }
    }

    let mut trellis_keys: Vec<&String> = obj.keys().filter(|k| k.ends_with(".trellis")).collect();
    trellis_keys.sort();
    println!(
        "file={} trellis_tensors={} markers={:?}",
        path,
        trellis_keys.len(),
        markers.values().collect::<Vec<_>>()
    );
    println!(
        "{:<58} {:>4} {:>3} {:>7} {:>12} {:>12} {:>12} {:>12} {:>7}",
        "tensor", "K", "cb", "tiles", "mean", "std", "min", "max", "finite%"
    );

    let mut grand_tiles = 0usize;
    let mut grand_vals = 0usize;
    let mut grand_nonfinite = 0usize;
    for key in trellis_keys {
        let e = &obj[key];
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape.len(), 3, "{key}: trellis must be 3-D");
        let (a, b, c) = (shape[0], shape[1], shape[2]);
        // Packed tile = 16*K u16 (+8 for half-integer mul1 rates, unsupported here).
        assert_eq!(c % 16, 0, "{key}: fractional-K tiles are Phase-1 scope");
        let k = c / 16;
        let bits = k as u32;
        let prefix = key.strip_suffix(".trellis").unwrap();
        let cb = markers.get(prefix).copied().unwrap_or(Codebook::Cb0);

        let off0 = e["data_offsets"][0].as_u64().unwrap() as usize;
        let off1 = e["data_offsets"][1].as_u64().unwrap() as usize;
        let ntiles = a * b;
        let words_per_tile_u16 = c;
        assert_eq!((off1 - off0) / 2, ntiles * words_per_tile_u16);

        let sample = max_tiles.min(ntiles);
        f.seek(SeekFrom::Start((data_start + off0) as u64))?;
        let mut raw = vec![0u8; sample * words_per_tile_u16 * 2];
        f.read_exact(&mut raw)?;
        let u16s: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();

        // Same bytes the CUDA kernel reads as uint32_t*: LE pair packing.
        let mut mean = 0f64;
        let mut m2 = 0f64;
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut n = 0usize;
        let mut nonfinite = 0usize;
        for t in 0..sample {
            let tile16 = &u16s[t * words_per_tile_u16..(t + 1) * words_per_tile_u16];
            let tile32: Vec<u32> = tile16
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0] as u32) | ((p[1] as u32) << 16))
                .collect();
            for v in decode_tile(cb, bits, &tile32) {
                n += 1;
                if !v.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                min = min.min(v);
                max = max.max(v);
                let d = v as f64 - mean;
                mean += d / (n - nonfinite) as f64;
                m2 += d * (v as f64 - mean);
            }
        }
        let std = if n > nonfinite {
            (m2 / (n - nonfinite) as f64).sqrt()
        } else {
            f64::NAN
        };
        grand_tiles += sample;
        grand_vals += n;
        grand_nonfinite += nonfinite;
        println!(
            "{:<58} {:>4} {:>3} {:>7} {:>12.5} {:>12.5} {:>12.5} {:>12.5} {:>6.2}%",
            key,
            format!("{k}"),
            match cb {
                Codebook::Cb0 => "cb0",
                Codebook::Cb1 => "cb1",
                Codebook::Cb2 => "cb2",
            },
            ntiles,
            mean,
            std,
            min,
            max,
            100.0 * (n - nonfinite) as f64 / n as f64,
        );
    }
    println!("sampled_tiles={grand_tiles} values={grand_vals} nonfinite={grand_nonfinite}");
    if grand_nonfinite > 0 {
        anyhow::bail!("non-finite decodes from real checkpoint bytes");
    }
    Ok(())
}
