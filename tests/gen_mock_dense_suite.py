"""Mock inputs and par files for the dense-W tool regression against Masaaki's.

    python3 gen_mock_dense_suite.py <run dir>

One set of random fields serves every tool. Each field is written twice, as
one (nt, rows, cols) file for Masaaki's tools and as timeSliceIO for ours, so
the comparison also exercises both of our reader paths. Written under <run dir>:

    in_single/, in_ts/            the fields
    par.product.{ref,test}.xml    A2AMesonFieldProduct / A2APixPi
    par.flexible.{ref,test}.xml   FlexibleContractorDWC / FlexibleContractorDense
    par.bubble.{ref,test}.xml     BubbleContractorDWC / BubbleContractorDense
    manifest.json                which reference output each of ours is checked against

Products are written by both product tools (ref/ and test/); each contractor
then reads the products of its own author, which the product diff has already
shown to be identical. Entries are random rather than physical: this checks
index bookkeeping (dense remap, offsets, orientation, precision, layout), not
physics. Run everything with run_mock_dense_suite.sh.
"""
import json
import os
import sys

import h5py
import numpy as np

NT, NSC = 8, 12
out = os.path.abspath(sys.argv[1])
rng = np.random.default_rng(20261110)


def dense(nlow, nhit):
    return nlow + nhit * NSC


def expanded(nlow, nhit):
    return nlow + nhit * NT * NSC


def to_compound(a, real):
    c = np.empty(a.shape, dtype=np.dtype([("re", real), ("im", real)]))
    c["re"], c["im"] = a.real, a.imag
    return c


def write_single(path, group, a, real):
    with h5py.File(path, "w") as f:
        f.create_group(group).create_dataset("a2aMatrix", data=to_compound(a, real))


def write_ts(stem, group, a, real):
    for t in range(a.shape[0]):
        with h5py.File(f"{stem}.t{t:04d}.h5", "w") as f:
            f.create_group(group).create_dataset("a2aMatrix", data=to_compound(a[t:t+1], real))


def b(x):
    return str(x).lower()


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


def field_dims(name):
    rd, cd, nlow, nhit, _ = FIELDS[name]
    return (dense if rd else expanded)(nlow, nhit), (dense if cd else expanded)(nlow, nhit)


def axes(name):
    """(row, col) as (dense, nLow, timeOffset) for a field or a product."""
    if name in PRODUCTS:
        left, right, dl, dr = PRODUCTS[name]
        lr, _, ln, _, _ = FIELDS[left]
        _, rc, rn, _, _ = FIELDS[right]
        return (lr, ln, dl % NT), (rc, rn, dr % NT)
    rd, cd, nlow, _, _ = FIELDS[name]
    return (rd, nlow, 0), (cd, nlow, 0)


def test_name(name):
    if name in PRODUCTS:
        return name
    return name + ("_sf" if name in SINGLE_FILE_INPUTS else "_ts")


def test_matrix(name, ts, path):
    def axis(a):
        return f"<dense>{b(a[0])}</dense><nLow>{a[1]}</nLow><timeOffset>{a[2]}</timeOffset>"
    row, col = axes(name)
    tag = name if name in PRODUCTS else name + ("_ts" if ts else "_sf")
    return (f"<elem><name>{tag}</name><file>{path}</file><dataset>{name}</dataset>"
            f"<timeSliceIO>{b(ts)}</timeSliceIO><row>{axis(row)}</row><col>{axis(col)}</col></elem>")


def ref_matrix(name, path):
    row, col = axes(name)
    return (f"<elem><file>{path}</file><dataset>{name}</dataset><cacheSize>{NT}</cacheSize>"
            f"<name>{name}</name><initDenseRow>{row[1] if row[0] else -1}</initDenseRow>"
            f"<initDenseCol>{col[1] if col[0] else -1}</initDenseCol>"
            f"<timeOffsetRow>{row[2]}</timeOffsetRow><timeOffsetCol>{col[2]}</timeOffsetCol></elem>")


def ref_stem(terms):
    return "_".join(f"{t}_{'x' if s else 'y'}{o}" for t, s, o in terms)


def grid(body):
    return f'<?xml version="1.0"?><grid>{body}</grid>'


def traj():
    return "<trajCounter><start>0</start><end>1</end><step>1</step></trajCounter>"


def product_xml_ref():
    m = "".join(f"<elem><name>{n}</name><file>{out}/in_single/{n}.h5</file><group>{n}</group></elem>"
                for n in FIELDS)
    p = ""
    for n, (l, r, dl, dr) in PRODUCTS.items():
        (_, _, _), lcol = axes(l)
        rrow, _ = axes(r)
        p += (f"<elem><mfL>{l}</mfL><mfR>{r}</mfR><deltaL>{dl}</deltaL><deltaR>{dr}</deltaR>"
              f"<ifDenseL>{b(lcol[0])}</ifDenseL><ifDenseR>{b(rrow[0])}</ifDenseR>"
              f"<initDense>{FIELDS[l][2]}</initDense><output>{out}/ref/{n}.h5</output>"
              f"<outputGroup>{n}</outputGroup></elem>")
    return grid(f"<global>{traj()}</global><a2aMatrix>{m}</a2aMatrix><product>{p}</product>")


def product_xml_test():
    m = "".join(test_matrix(n, ts, f"{out}/{'in_ts' if ts else 'in_single'}/{n}.h5")
                for n in FIELDS for ts in (True, False))
    p = "".join(f"<elem><left>{test_name(l)}</left><right>{r}_ts</right><deltaL>{dl}</deltaL>"
                f"<deltaR>{dr}</deltaR><output>{out}/test/{n}.h5</output><dataset>{n}</dataset></elem>"
                for n, (l, r, dl, dr) in PRODUCTS.items())
    return grid(f"<global>{traj()}<nt>{NT}</nt></global><a2aMatrix>{m}</a2aMatrix>"
                f"<product>{p}</product>")


def used_names():
    return sorted({t for terms, *_ in CONTRACTIONS.values() for t, _, _ in terms})


def flexible_xml_ref():
    m = "".join(ref_matrix(n, f"{out}/{'ref' if n in PRODUCTS else 'in_single'}/{n}.h5")
                for n in used_names())
    p = ""
    for terms, tr, lo, hi, avg in CONTRACTIONS.values():
        t = "".join(f"<elem><term>{n}</term><tdps>{b(s)}</tdps><time>{o}</time></elem>"
                    for n, s, o in terms)
        p += (f"<elem><terms>{t}</terms><translations>{tr}</translations><min_t>{lo}</min_t>"
              f"<max_t>{hi}</max_t><translationAverage>{b(avg)}</translationAverage></elem>")
    return grid(f"<global>{traj()}<nt>{NT}</nt><diskVectorDir>{out}/dv_ref</diskVectorDir>"
                f"<output>{out}/corr_ref</output></global><a2aMatrix>{m}</a2aMatrix>"
                f"<product>{p}</product>")


def flexible_xml_test():
    m = ""
    for n in used_names():
        if n in PRODUCTS:
            m += test_matrix(n, True, f"{out}/test/{n}.h5")
        else:
            ts = n not in SINGLE_FILE_INPUTS
            m += test_matrix(n, ts, f"{out}/{'in_ts' if ts else 'in_single'}/{n}.h5")
    c = ""
    for name, (terms, tr, lo, hi, avg) in CONTRACTIONS.items():
        t = "".join(f"<elem><name>{test_name(n)}</name><role>{'sink' if s else 'source'}</role>"
                    f"<offset>{o}</offset></elem>" for n, s, o in terms)
        c += (f"<elem><name>{name}</name><terms>{t}</terms><translations>{tr}</translations>"
              f"<min_t>{lo}</min_t><max_t>{hi}</max_t>"
              f"<translationAverage>{b(avg)}</translationAverage></elem>")
    return grid(f"<global>{traj()}<nt>{NT}</nt><output>{out}/corr_test</output></global>"
                f"<a2aMatrix>{m}</a2aMatrix><contraction>{c}</contraction>")


def bubble_ref_terms(terms):
    """Masaaki's (terms, times) for one of our bubbles, term 0 moved to dt."""
    o0 = terms[0][1]
    return [n for n, _ in terms], [(o - o0) % NT for _, o in terms[1:]]


def bubble_stem(terms):
    names, times = bubble_ref_terms(terms)
    return "".join(f"{n}_{t}_" for n, t in zip(names, times)) + names[-1]


def bubble_names():
    return sorted({n for terms in BUBBLES.values() for n, _ in terms})


def bubble_xml_ref():
    m = "".join(ref_matrix(n, f"{out}/{'ref' if n in PRODUCTS else 'in_single'}/{n}.h5")
                for n in bubble_names())
    p, seen = "", set()
    for terms in BUBBLES.values():
        if bubble_stem(terms) in seen:
            continue
        seen.add(bubble_stem(terms))
        names, times = bubble_ref_terms(terms)
        t = "".join(f"<elem>{x}</elem>" for x in times)
        p += f"<elem><terms>{' '.join(names)}</terms><times>{t}</times></elem>"
    return grid(f"<global>{traj()}<nt>{NT}</nt><diskVectorDir>{out}/dv_bub_ref</diskVectorDir>"
                f"<output>{out}/bub_ref</output></global><a2aMatrix>{m}</a2aMatrix>"
                f"<product>{p}</product>")


def bubble_xml_test():
    m = ""
    for n in bubble_names():
        if n in PRODUCTS:
            m += test_matrix(n, True, f"{out}/test/{n}.h5")
        else:
            ts = n not in SINGLE_FILE_INPUTS
            m += test_matrix(n, ts, f"{out}/{'in_ts' if ts else 'in_single'}/{n}.h5")
    c = ""
    for name, terms in BUBBLES.items():
        t = "".join(f"<elem><name>{test_name(n)}</name><offset>{o}</offset></elem>"
                    for n, o in terms)
        c += f"<elem><name>{name}</name><terms>{t}</terms></elem>"
    return grid(f"<global>{traj()}<nt>{NT}</nt><output>{out}/bub_test</output></global>"
                f"<a2aMatrix>{m}</a2aMatrix><bubble>{c}</bubble>")


def expand_translations(tr):
    ts = []
    for tok in tr.split():
        a, _, z = tok.partition("..")
        ts += list(range(int(a), int(z or a) + 1))
    return ts


for d in ("in_single", "in_ts", "ref", "test"):
    os.makedirs(os.path.join(out, d), exist_ok=True)

for name, (*_, real) in FIELDS.items():
    ni, nj = field_dims(name)
    a = rng.standard_normal((NT, ni, nj)) + 1j * rng.standard_normal((NT, ni, nj))
    write_single(os.path.join(out, "in_single", name + ".h5"), name, a, real)
    write_ts(os.path.join(out, "in_ts", name), name, a, real)

for fn, text in (("par.product.ref.xml", product_xml_ref()),
                 ("par.product.test.xml", product_xml_test()),
                 ("par.flexible.ref.xml", flexible_xml_ref()),
                 ("par.flexible.test.xml", flexible_xml_test()),
                 ("par.bubble.ref.xml", bubble_xml_ref()),
                 ("par.bubble.test.xml", bubble_xml_test())):
    with open(os.path.join(out, fn), "w") as f:
        f.write(text)

manifest = {
    "nt": NT,
    "contractions": [
        {"test": name, "ref": ref_stem(terms), "average": avg,
         "translations": expand_translations(tr), "min_t": lo, "max_t": hi}
        for name, (terms, tr, lo, hi, avg) in CONTRACTIONS.items()
    ],
    "bubbles": [
        {"test": name, "ref": bubble_stem(terms), "roll": terms[0][1]}
        for name, terms in BUBBLES.items()
    ],
    "identities": IDENTITIES,
}
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

print(f"wrote {len(FIELDS)} fields, {len(PRODUCTS)} products, "
      f"{len(CONTRACTIONS)} contractions, {len(BUBBLES)} bubbles under {out}")
