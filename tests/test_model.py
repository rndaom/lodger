"""The model file reader: sizes of every piece from the header alone, the layers llama.cpp skips, what each layer
keeps for the context."""
import os
import struct
import tempfile
import unittest

from lodger import model

MiB = 1 << 20


def _s(t):
    b = t.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v):
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", 4) + struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<I", 6) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _s(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(x) for x in v)
    if isinstance(v, list) and all(isinstance(x, bool) for x in v):
        return struct.pack("<IIQ", 9, 7, len(v)) + b"".join(struct.pack("<?", x) for x in v)
    return struct.pack("<IIQ", 9, 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)


def write_gguf(path, kv, tensors, align=32):
    """A GGUF file: settings `kv`, then `tensors` as (name, ggml type, shape, bytes it takes in the file), each
    padded to `align`. The data is zeros (a sparse file), like a header stub."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1)
    head += _s("general.alignment") + _value(align)
    for k, v in kv.items():
        head += _s(k) + _value(v)
    offset = 0
    for name, kind, shape, size in tensors:
        head += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        head += struct.pack("<IQ", kind, offset)
        offset += (size + align - 1) // align * align
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + offset)
    return path


def q4_0(rows, cols):
    """(ggml type, shape, bytes) of a Q4_0 matrix: 18 bytes per 32 numbers."""
    return 2, [cols, rows], rows * cols // 32 * 18


def f32(*shape):
    n = 1
    for d in shape:
        n *= d
    return 0, list(shape), n * 4


def layer(i, kind="attn", cols=256, rows=256, experts=0):
    """The tensors of one layer: attention or recurrent (ssm), an FFN, and `experts` experts."""
    out = []
    if kind == "attn":
        out += [(f"blk.{i}.attn_q.weight", *q4_0(rows, cols)), (f"blk.{i}.attn_k.weight", *q4_0(rows, cols))]
    else:
        out += [(f"blk.{i}.attn_qkv.weight", *q4_0(rows, cols)), (f"blk.{i}.ssm_out.weight", *q4_0(rows, cols))]
    out += [(f"blk.{i}.attn_norm.weight", *f32(cols)), (f"blk.{i}.ffn_up.weight", *q4_0(rows, cols))]
    if experts:
        out += [(f"blk.{i}.ffn_up_exps.weight", 2, [cols, rows, experts], rows * cols // 32 * 18 * experts),
                (f"blk.{i}.ffn_gate_inp.weight", *f32(cols, experts)),
                (f"blk.{i}.ffn_norm_exps.weight", *f32(cols, experts))]
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lodger-model-")
        self.addCleanup(lambda: [os.remove(os.path.join(self.dir, n)) for n in os.listdir(self.dir)]
                        and os.rmdir(self.dir))

    def file(self, kv, tensors, name="m.gguf", **kw):
        kv = {"general.architecture": "toy", "general.name": "Toy", **kv}
        return write_gguf(os.path.join(self.dir, name), kv, tensors, **kw)


class ReaderTest(Base):
    def test_sizes_come_from_type_and_shape(self):
        t = [("token_embd.weight", *q4_0(1000, 256)), ("blk.0.attn_norm.weight", *f32(256)),
             ("blk.0.attn_q.weight", 12, [256, 64], 64 * 144), ("output.weight", 14, [256, 1000], 1000 * 210)]
        kv, tensors = model.read_gguf(self.file({}, t))
        self.assertEqual([(n, b) for n, b, _, _ in tensors], [(n, b) for n, _, _, b in t])

    def test_a_type_lodger_doesnt_know_takes_the_room_to_the_next_tensor(self):
        t = [("blk.0.attn_q.weight", 142, [256, 64], 5000), ("blk.0.attn_k.weight", *q4_0(64, 256))]
        _, tensors = model.read_gguf(self.file({}, t))
        self.assertEqual(tensors[0][1], 5024)                     # 5000, padded to 32
        self.assertEqual(tensors[1][1], q4_0(64, 256)[2])

    def test_settings_short_lists_kept_long_ones_skipped(self):
        kv = {"toy.block_count": 2, "toy.attention.sliding_window_pattern": [True, False],
              "tokenizer.ggml.tokens": [f"t{i}" for i in range(5000)], "toy.rope.freq_base": 1e6}
        kv_read, _ = model.read_gguf(self.file(kv, [("blk.0.attn_q.weight", *q4_0(32, 32))]))
        self.assertEqual(kv_read["toy.block_count"], 2)
        self.assertEqual(kv_read["toy.attention.sliding_window_pattern"], [True, False])
        self.assertNotIn("tokenizer.ggml.tokens", kv_read)
        self.assertEqual(kv_read["general.architecture"], "toy")

    def test_not_gguf_is_a_plain_error(self):
        p = os.path.join(self.dir, "x.gguf")
        with open(p, "wb") as f:
            f.write(b"<html>" * 10)
        with self.assertRaises(ValueError):
            model.read_gguf(p)
        with open(p, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + _s("general.name"))     # cut short
        with self.assertRaises(ValueError):
            model.read_gguf(p)


class ModelTest(Base):
    def test_pieces_are_grouped_for_the_planner(self):
        t = [("token_embd.weight", *q4_0(1000, 256)), ("output_norm.weight", *f32(256)),
             ("output.weight", *q4_0(1000, 256)), ("rope_freqs.weight", *f32(64))]
        t += layer(0, experts=4) + layer(1, experts=4)
        m = model.Model(self.file({"toy.block_count": 2, "toy.context_length": 4096}, t))
        self.assertEqual(m.n_layer, 2)
        self.assertEqual(m.token_embd, q4_0(1000, 256)[2])
        self.assertEqual(m.output, q4_0(1000, 256)[2] + 1024)
        self.assertEqual(m.other, 256)
        self.assertEqual(m.layers[0]["experts"], 256 * 256 // 32 * 18 * 4)
        # only up/down/gate experts are what --n-cpu-moe moves (llama.cpp's LLM_FFN_EXPS_REGEX); the experts' norm stays
        self.assertEqual(m.layers[0]["dense"], 3 * q4_0(256, 256)[2] + 1024 + 2 * 256 * 4 * 4)
        self.assertFalse(m.tied)
        self.assertEqual(m.meta["ctx_train"], 4096)
        self.assertTrue(m.is_moe)

    def test_tied_output_is_a_copy_of_the_embeddings(self):
        t = [("token_embd.weight", *q4_0(1000, 256))] + layer(0)
        m = model.Model(self.file({"toy.block_count": 1}, t))
        self.assertTrue(m.tied)
        self.assertEqual(m.output, m.token_embd)
        self.assertEqual(m.repacked({2: 8}, set(), set(), True), (m.token_embd, m.token_embd))

    def test_multi_token_prediction_layers_are_not_loaded(self):
        # llama.cpp skips the nextn layers (TENSOR_SKIP) unless asked to load them for MTP drafting; they still
        # count in -ngl's layer numbers
        t = layer(0) + layer(1) + layer(2) + [("blk.2.nextn.eh_proj.weight", *q4_0(256, 512))]
        m = model.Model(self.file({"toy.block_count": 3, "toy.nextn_predict_layers": 1}, t))
        self.assertEqual(m.n_layer, 3)
        self.assertEqual(m.layers[2], {"dense": 0, "experts": 0})
        self.assertGreater(m.mtp[2]["dense"], 0)
        self.assertIsNone(m.kinds[2])
        self.assertEqual(m.repacked({2: 8}, {0, 1, 2}, set(), False)[0], 2 * 3 * q4_0(256, 256)[2])

    def test_mtp_drafting_loads_the_mtp_layers(self):
        t = layer(0) + layer(1) + layer(2) + [("blk.2.nextn.eh_proj.weight", *q4_0(256, 512))]
        m = model.Model(self.file({"toy.block_count": 3, "toy.nextn_predict_layers": 1}, t))
        d = m.with_mtp()
        self.assertEqual(d.layers[2], m.mtp[2])
        self.assertEqual(d.total, m.total + m.mtp[2]["dense"])
        self.assertEqual(m.layers[2], {"dense": 0, "experts": 0})          # the model itself is unchanged
        self.assertEqual(d.repacked({2: 8}, {0, 1, 2}, set(), False)[0], 3 * 3 * q4_0(256, 256)[2] + q4_0(256, 512)[2])
        self.assertEqual((d.mtp, d.kinds[2]), (m.mtp, None))

    def test_layer_count_is_llamas_even_without_the_last_layers_tensors(self):
        # a trunk-only file of a model with an MTP layer: llama.cpp still counts block_count layers
        m = model.Model(self.file({"toy.block_count": 3, "toy.nextn_predict_layers": 1}, layer(0) + layer(1)))
        self.assertEqual(m.n_layer, 3)

    def test_what_each_layer_keeps_for_the_context(self):
        t = layer(0, "recr") + layer(1, "recr") + layer(2) + layer(3)
        m = model.Model(self.file({"toy.block_count": 4}, t))
        self.assertEqual([m.kinds[i] for i in range(4)], ["recr", "recr", "attn", "attn"])
        kv = {"toy.block_count": 4, "toy.attention.sliding_window": 512}
        m = model.Model(self.file({**kv, "toy.attention.sliding_window_pattern": [True, True, False, True]},
                                  layer(0) + layer(1) + layer(2) + layer(3), name="swa.gguf"))
        self.assertEqual([m.kinds[i] for i in range(4)], ["swa", "swa", "attn", "swa"])
        m = model.Model(self.file({**kv, "toy.attention.sliding_window_pattern": 2},
                                  layer(0) + layer(1) + layer(2) + layer(3), name="swa2.gguf"))
        self.assertEqual([m.kinds[i] for i in range(4)], ["swa", "attn", "swa", "attn"])
        # a sliding window, but not which layers: every attention layer counts the same (as before)
        m = model.Model(self.file(kv, layer(0) + layer(1), name="swa3.gguf"))
        self.assertEqual([m.kinds[i] for i in range(2)], ["attn", "attn"])

    def test_repacked_weights(self):
        t = [("token_embd.weight", *q4_0(1000, 256)), ("output.weight", *q4_0(1000, 256))]
        t += layer(0, experts=4) + layer(1, experts=4) + [("blk.1.attn_v.weight", *q4_0(12, 256))]  # 12 rows: not 8n
        m = model.Model(self.file({"toy.block_count": 2}, t))
        one = q4_0(256, 256)[2]
        self.assertEqual(m.repacked({}, {0, 1}, {0, 1}, True), (0, 0))
        self.assertEqual(m.repacked({2: 8}, {0}, set(), False), (3 * one, 0))
        self.assertEqual(m.repacked({2: 8}, set(), {1}, False), (4 * one, 0))
        self.assertEqual(m.repacked({2: 4}, {1}, set(), True), (3 * one + q4_0(12, 256)[2] + q4_0(1000, 256)[2],
                                                                q4_0(1000, 256)[2]))

    def test_read_once_per_change(self):
        p = self.file({"toy.block_count": 1}, layer(0))
        a = model.load(p)
        self.assertIs(model.load(p), a)
        self.file({"toy.block_count": 2}, layer(0) + layer(1))
        os.utime(p, ns=(1, 1))
        b = model.load(p)
        self.assertIsNot(b, a)
        self.assertEqual(b.n_layer, 2)


if __name__ == "__main__":
    unittest.main()
