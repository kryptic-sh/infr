"""Generate compact, independently rounded ROW256 goldens from the pinned E4B checkpoint."""
import hashlib
import json
import struct
import sys
from pathlib import Path

PIN = "b5fae5c635b94490fc940818767afa5c390d39b4"
SHA256 = "b1685ab35aca62ae1cce488e9fbb005f3a3c372217fc7f40854c358ad80e7994"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def decode(group, signs):
    gain = struct.unpack("<e", group[:2])[0]
    inv = struct.unpack("<e", struct.pack("<H", 0x1EEE))[0]
    bias = struct.unpack("<e", struct.pack("<H", 0xC931))[0]
    values = []
    for i in range(256):
        word = 0
        for b in range(16):
            p = ((i - b // 4) % 256) * 4 + b % 4
            word |= ((group[2 + p // 8] >> (p % 8)) & 1) << b
        h = (word * 0x83DCD12D) & 0xFFFFFFFF
        total = 1024 + sum(h.to_bytes(4, "little"))
        values.append(f32(half(total * inv + bias) * gain))
    stride = 1
    while stride < 256:
        previous = values[:]
        for i in range(256):
            peer = previous[i ^ stride]
            values[i] = f32(peer - previous[i] if i & stride else previous[i] + peer)
        stride *= 2
    return b"".join(struct.pack("<e", f32(f32(value / 16) * sign)) for value, sign in zip(values, struct.unpack("<256e", signs)))


def main():
    path = Path(sys.argv[1])
    with path.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == SHA256, "checkpoint hash mismatch"
    entries = []
    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))
        start = 8 + size
        for prefix in ["model.language_model.embed_tokens", "model.language_model.embed_tokens_per_layer"]:
            trellis = header[prefix + ".trellis"]
            signs = header[prefix + ".signs"]
            groups = signs["shape"][0]
            row_bytes = trellis["shape"][1] * 2
            selected_signs = b""
            for group in [0, groups - 1]:
                file.seek(start + signs["data_offsets"][0] + group * 512)
                selected_signs += file.read(512)
            packed = b""
            expected = b""
            for token in [0, 106, 262143]:
                for part, group in enumerate([0, groups - 1]):
                    file.seek(start + trellis["data_offsets"][0] + token * row_bytes + group * 130)
                    data = file.read(130)
                    packed += data
                    expected += decode(data, selected_signs[part * 512:(part + 1) * 512])
            entries.append({"name": prefix, "source_tokens": [0, 106, 262143], "source_groups": [0, groups - 1], "packed": packed.hex(), "signs": selected_signs.hex(), "expected_f16": expected.hex()})
    output = Path(__file__).parent / "fixtures" / "exl3_embedding_rows.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": "cortexist/gemma-4-E4B-it-qat-exl3-4.0bpw", "revision": PIN, "sha256": SHA256, "entries": entries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
