use super::{bda_weight_block_size, BDA_BLOCK_MIN, BDA_WEIGHT_ALIGN};

fn reserved(sizes: &[u64], capacity: impl Fn(u64) -> u64) -> u64 {
    let mut blocks: Vec<(u64, u64)> = Vec::new();
    for &size in sizes {
        let want = size.max(1).next_multiple_of(BDA_WEIGHT_ALIGN);
        if let Some((_, cursor)) = blocks
            .iter_mut()
            .find(|(size, cursor)| *cursor + want <= *size)
        {
            *cursor += want;
        } else {
            let size = capacity(want);
            assert!(size >= want);
            blocks.push((size, want));
        }
    }
    blocks.iter().map(|(size, _)| size).sum()
}

#[test]
fn large_half_block_weights_reserve_exactly() {
    let bytes = 3584 * 18944 * 4 / 8;
    assert_eq!(bytes, 33_947_648);
    assert_eq!(bda_weight_block_size(bytes, u64::MAX), bytes);
    assert_eq!(
        reserved(&[bytes; 84], |want| bda_weight_block_size(want, u64::MAX)),
        bytes * 84
    );
}

#[test]
fn threshold_preserves_small_tensor_batching() {
    let half = BDA_BLOCK_MIN / 2;
    assert_eq!(
        bda_weight_block_size(half - BDA_WEIGHT_ALIGN, u64::MAX),
        BDA_BLOCK_MIN
    );
    assert_eq!(bda_weight_block_size(half, u64::MAX), half);
    assert_eq!(
        bda_weight_block_size(half + BDA_WEIGHT_ALIGN, u64::MAX),
        half + BDA_WEIGHT_ALIGN
    );
    assert_eq!(
        bda_weight_block_size(BDA_BLOCK_MIN, u64::MAX),
        BDA_BLOCK_MIN
    );
    assert_eq!(
        reserved(&[1, 257, 4096, 1 << 20], |want| bda_weight_block_size(
            want,
            u64::MAX
        )),
        BDA_BLOCK_MIN
    );
}

#[test]
fn block_capacity_respects_allocation_limit() {
    let limit = 16 << 20;
    assert_eq!(bda_weight_block_size(8 << 20, limit), limit);
    assert_eq!(bda_weight_block_size(40 << 20, limit), limit);
    assert!(bda_weight_block_size(40 << 20, limit) < 40 << 20);
}

#[test]
fn qwen25_native_weight_plan_avoids_gibibytes_of_unused_tails() {
    let k = 3584_u64;
    let ff = 18944_u64;
    let vocab = 152064_u64;
    let projection = |sizes: &mut Vec<u64>, input: u64, output: u64, bits: u64| {
        sizes.extend([input * output * bits / 8, input * 2, output * 2]);
    };
    let mut sizes = Vec::new();
    for _ in 0..28 {
        sizes.extend([k * 4, k * 4]);
        for output in [k, 512, 512] {
            projection(&mut sizes, k, output, 4);
            sizes.push(output * 4);
        }
        projection(&mut sizes, k, k, 4);
        projection(&mut sizes, k, ff, 4);
        projection(&mut sizes, k, ff, 4);
        projection(&mut sizes, ff, k, 4);
    }
    sizes.push(k * 4);
    projection(&mut sizes, k, vocab, 6);
    sizes.push(k * vocab * 2);
    let old = reserved(&sizes, |want| want.max(BDA_BLOCK_MIN));
    let new = reserved(&sizes, |want| bda_weight_block_size(want, u64::MAX));
    let payload: u64 = sizes
        .iter()
        .map(|size| size.next_multiple_of(BDA_WEIGHT_ALIGN))
        .sum();
    assert!(old - new > 2 << 30, "old={old}, new={new}");
    assert!(new >= payload);
    assert!(
        new - payload < BDA_BLOCK_MIN,
        "new={new}, payload={payload}"
    );
}
