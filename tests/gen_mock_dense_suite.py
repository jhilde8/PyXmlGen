"""Mock inputs and par files for the dense-W tool regression against Masaaki's.

    python3 gen_mock_dense_suite.py <run dir>

One set of random fields serves every tool. Each field is written twice, as
one (nt, rows, cols) file for Masaaki's tools and as timeSliceIO for ours, so
the comparison also exercises both of our reader paths. Written under <run dir>:

    in_single/, in_ts/            the fields
    par.product.{ref,test}.xml    A2AMesonFieldProduct / A2APixPi
    par.flexible.{ref,test}.xml   FlexibleContractorDWC / FlexibleContractorDense
    par.bubble.{ref,test}.xml     BubbleContractorDWC / BubbleContractorDense
    manifest.json                 which reference output each of ours is checked against

Our par files come from contractor_xml, the production toolkit, and Masaaki's
from tests/dwc_xml. Products are written by both product tools (ref/ and
test/); each contractor then reads the products of its own author, which the
product diff has already shown to be identical. Entries are random rather than
physical: this checks index bookkeeping (dense remap, offsets, orientation,
precision, layout), not physics. Run everything with run_mock_dense_suite.sh.
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contractor_xml import BubbleJob, FlexibleJob, PixPiJob, axis
from dwc_xml import BubbleDWCJob, FlexibleDWCJob, ProductJob

NT, NSC = 8, 12
TRAJ = (0, 1)
out = os.path.abspath(sys.argv[1])
rng = np.random.default_rng(20261110)

# name: (rowDense, colDense, nLow, nHit, real). Both axes share nLow and nHit.
FIELDS = {
    "wv_l":   (True,  False, 5, 2, "<f8"),
    "wv_l_f": (True,  False, 5, 2, "<f4"),
    "wv_s":   (True,  False, 0, 2, "<f8"),
    "vw_l":   (False, True,  5, 2, "<f8"),
    "wv_a":   (True,  False, 5, 1, "<f8"),
    "vv_l":   (False, False, 5, 2, "<f8"),
    "ww_l":   (True,  True,  5, 2, "<f8"),
}

# name: (left, right, deltaL, deltaR). Single-file left inputs where marked.
PRODUCTS = {
    "p_wv_dR2":  ("wv_l",   "wv_l", 0, 2),
    "p_wv_dL3":  ("wv_l",   "wv_l", 3, 0),
    "p_vw_dR1":  ("vw_l",   "vw_l", 0, 1),
    "p_s_dR1":   ("wv_s",   "wv_s", 0, 1),
    "p_mixFD":   ("wv_l_f", "wv_l", 0, 1),
    "p_ord":     ("wv_l",   "vw_l", 0, 0),
    "p_ama_dR5": ("wv_a",   "wv_a", 0, 5),
}
SINGLE_FILE_INPUTS = {"wv_l_f", "vw_l"}

# name: ([(term, sink, offset)], translations, min_t, max_t, translationAverage)
CONTRACTIONS = {
    "c_wv":    ([("wv_l", False, 0), ("wv_l", True, 0)], "0..7", 0, 7, False),
    "c_s":     ([("wv_s", False, 0), ("wv_s", True, 0)], "0..7", 0, 7, True),
    "c_f":     ([("wv_l_f", False, 0), ("wv_l", True, 0)], "0..7", 0, 7, False),
    "c_ama":   ([("wv_a", False, 0), ("wv_a", True, 0)], "1 3..5", 1, 6, False),
    "c_vw":    ([("vw_l", False, 0), ("vw_l", True, 1)], "0..7", 0, 7, False),
    "c_vvww":  ([("vv_l", False, 0), ("ww_l", True, 0)], "0..7", 0, 7, False),
    "c_wwvv":  ([("ww_l", False, 0), ("vv_l", True, 0)], "0..7", 0, 7, False),
    "c_rect":  ([("p_wv_dR2", False, NT - 2), ("p_wv_dR2", True, 0)], "0..7", 0, 7, False),
    "c_cross": ([("wv_l", False, 0), ("wv_l", True, 0), ("wv_l", False, 2), ("wv_l", True, 2)],
                "0..7", 1, 5, False),
}

# name: [(term, offset)]. Masaaki's term 0 sits at dt, so ours with a nonzero
# first offset o0 is his output rolled by -o0.
BUBBLES = {
    "b_wv":     [("wv_l", 0)],
    "b_vw":     [("vw_l", 0)],
    "b_vv":     [("vv_l", 0)],
    "b_s":      [("wv_s", 0)],
    "b_f":      [("wv_l_f", 0)],
    "b_ama":    [("wv_a", 0)],
    "b_prod":   [("p_wv_dR2", 0)],
    "b_prodL":  [("p_wv_dL3", 0)],
    "b_prodvw": [("p_vw_dR1", 0)],
    "b_prodnz": [("p_wv_dR2", 6)],
    "b_pair":   [("wv_l", 0), ("wv_l", 2)],
    "b_pairL":  [("wv_l", 3), ("wv_l", 0)],
    "b_vvww":   [("vv_l", 0), ("ww_l", 1)],
}

# Pairs that must agree: tr of an A2APixPi product against the 2-term trace
# of its factors.
IDENTITIES = [("b_prod", "b_pair"), ("b_prodL", "b_pairL")]


def field_axes(name):
    rd, cd, nlow, _, _ = FIELDS[name]
    return axis(rd, nlow), axis(cd, nlow)


def field_dims(name):
    rd, cd, nlow, nhit, _ = FIELDS[name]
    return tuple(nlow + nhit * (NSC if d else NT * NSC) for d in (rd, cd))


def to_compound(a, real):
    c = np.empty(a.shape, dtype=np.dtype([("re", real), ("im", real)]))
    c["re"], c["im"] = a.real, a.imag
    return c


def write_fields():
    for d in ("in_single", "in_ts", "ref", "test"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    for name, (*_, real) in FIELDS.items():
        ni, nj = field_dims(name)
        a = rng.standard_normal((NT, ni, nj)) + 1j * rng.standard_normal((NT, ni, nj))
        with h5py.File(f"{out}/in_single/{name}.h5", "w") as f:
            f.create_group(name).create_dataset("a2aMatrix", data=to_compound(a, real))
        for t in range(NT):
            with h5py.File(f"{out}/in_ts/{name}.t{t:04d}.h5", "w") as f:
                f.create_group(name).create_dataset("a2aMatrix", data=to_compound(a[t:t+1], real))


def base_name(name):
    """Our matrix name for a base field: which of its two copies it reads."""
    return name + ("_sf" if name in SINGLE_FILE_INPUTS else "_ts")


def add_base(job, name, single_file):
    row, col = field_axes(name)
    if single_file:
        job.add_matrix(f"{name}_sf", f"{out}/in_single/{name}.h5", name, row, col,
                       time_slice_io=False)
    else:
        job.add_matrix(f"{name}_ts", f"{out}/in_ts/{name}.h5", name, row, col)


def products_test():
    """A2APixPi on both copies of every field; returns each product's
    a2aMatrix entry for the contractor jobs."""
    job = PixPiJob(NT, *TRAJ)
    for name in FIELDS:
        add_base(job, name, single_file=False)
        add_base(job, name, single_file=True)
    entries = {}
    for name, (left, right, dl, dr) in PRODUCTS.items():
        entries[name] = job.add_product(base_name(left), f"{right}_ts", dl, dr,
                                        f"{out}/test/{name}.h5", name)
    job.write(f"{out}/par.product.test.xml")
    return entries


def products_ref():
    job = ProductJob(*TRAJ)
    for name in FIELDS:
        job.add_matrix(name, f"{out}/in_single/{name}.h5", name, *field_axes(name))
    for name, (left, right, dl, dr) in PRODUCTS.items():
        job.add_product(left, right, dl, dr, f"{out}/ref/{name}.h5", name)
    job.write(f"{out}/par.product.ref.xml")


def declare_test(job, names, entries):
    for name in names:
        if name in PRODUCTS:
            job.add_matrix(name, **entries[name])
        else:
            add_base(job, name, single_file=name in SINGLE_FILE_INPUTS)


def declare_ref(job, names, entries):
    for name in names:
        if name in PRODUCTS:
            e = entries[name]
            job.add_matrix(name, f"{out}/ref/{name}.h5", name, e["row"], e["col"])
        else:
            job.add_matrix(name, f"{out}/in_single/{name}.h5", name, *field_axes(name))


def test_term(name):
    return name if name in PRODUCTS else base_name(name)


def flexible(entries):
    used = sorted({t for terms, *_ in CONTRACTIONS.values() for t, _, _ in terms})
    test = FlexibleJob(NT, f"{out}/corr_test", *TRAJ)
    ref = FlexibleDWCJob(NT, f"{out}/dv_ref", f"{out}/corr_ref", *TRAJ)
    declare_test(test, used, entries)
    declare_ref(ref, used, entries)
    stems = {}
    for name, (terms, tr, lo, hi, avg) in CONTRACTIONS.items():
        test.add_contraction(name, [(test_term(t), "sink" if s else "source", o)
                                    for t, s, o in terms], tr, lo, hi, avg)
        stems[name] = ref.add_contraction(terms, tr, lo, hi, avg)
    test.write(f"{out}/par.flexible.test.xml")
    ref.write(f"{out}/par.flexible.ref.xml")
    return stems


def bubbles(entries):
    used = sorted({t for terms in BUBBLES.values() for t, _ in terms})
    test = BubbleJob(NT, f"{out}/bub_test", *TRAJ)
    ref = BubbleDWCJob(NT, f"{out}/dv_bub_ref", f"{out}/bub_ref", *TRAJ)
    declare_test(test, used, entries)
    declare_ref(ref, used, entries)
    stems = {}
    for name, terms in BUBBLES.items():
        test.add_bubble(name, [(test_term(t), o) for t, o in terms])
        o0 = terms[0][1]
        stems[name] = ref.add_bubble([t for t, _ in terms],
                                     [(o - o0) % NT for _, o in terms[1:]])
    test.write(f"{out}/par.bubble.test.xml")
    ref.write(f"{out}/par.bubble.ref.xml")
    return stems


def expand_translations(tr):
    ts = []
    for tok in tr.split():
        a, _, z = tok.partition("..")
        ts += list(range(int(a), int(z or a) + 1))
    return ts


write_fields()
products_ref()
entries = products_test()
flex_stems = flexible(entries)
bub_stems = bubbles(entries)

manifest = {
    "nt": NT,
    "contractions": [
        {"test": name, "ref": flex_stems[name], "average": avg,
         "translations": expand_translations(tr), "min_t": lo, "max_t": hi}
        for name, (terms, tr, lo, hi, avg) in CONTRACTIONS.items()
    ],
    "bubbles": [
        {"test": name, "ref": bub_stems[name], "roll": terms[0][1]}
        for name, terms in BUBBLES.items()
    ],
    "identities": IDENTITIES,
}
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

print(f"wrote {len(FIELDS)} fields, {len(PRODUCTS)} products, "
      f"{len(CONTRACTIONS)} contractions, {len(BUBBLES)} bubbles under {out}")
