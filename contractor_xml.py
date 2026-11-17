"""
XML construction for the standalone contractor utilities.

This is a different document from hadrons_xml.Job. A Hadrons job is
<grid><parameters/><modules/></grid> and is executed by the scheduler; a
utility par file is <grid><global/>...</grid> and is read straight into
Serializable structs by main(). Nothing is shared but the option-serialisation
helpers, so the two live side by side rather than one wrapping the other.

The contraction utilities all sit on Hadrons/utilities/A2ADenseContract.hpp:

    PixPiJob       A2APixPi                products C[t] = L[t+dL] R[t+dR]
    FlexibleJob    FlexibleContractorDense  N-term source/sink traces
    BubbleJob      BubbleContractorDense    1- and 2-term bubbles, every t
    K2pipiJob      K2pipiContractorDense    tr(fixed chain x swept field)

They share one <a2aMatrix> entry (MatrixPar): a name, the file and HDF5 group,
timeSliceIO, and a row and a col axis, each {dense, nLow, timeOffset}. Base
fields take their axes from config.field_axes(kind); a product's axes are
derived by PixPiJob.add_product, which returns the entry a contractor job
declares, so no offset is ever typed twice.

Element order follows the GRID_SERIALIZABLE_CLASS_MEMBERS declaration order in
the .cpp/.hpp. Grid's XmlReader looks members up by name, so order is not load
bearing, but a par file that reads top to bottom like the struct is easier to
check against it.

Every job checks on write() what the utility would refuse at run time and can
be known without the files: undeclared or duplicate names, delta and role
rules, time ranges. Dimension checks need the files and stay in the utility.
None of the three needs its output directory to exist.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from hadrons_xml import _add_option, _xml_scalar

# parseTimeRange's regex (A2ADenseContract.hpp): a bare timeslice or an
# inclusive a..b range, no sign.
TIME_RANGE = r"^([0-9]+|[0-9]+\.\.[0-9]+)$"


def _bool(parent, key, val):
    # Not via _add_option: its bool guard lists the Hadrons module options
    # that are genuinely bool, and these belong to the utilities' structs.
    ET.SubElement(parent, key).text = _xml_scalar(bool(val))


def _write(root, xml_path):
    xml_path = Path(xml_path)
    ET.indent(root, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=True)
    return xml_path


def _traj(parent, traj_start, traj_end, traj_step):
    traj = ET.SubElement(parent, "trajCounter")
    ET.SubElement(traj, "start").text = str(traj_start)
    ET.SubElement(traj, "end").text = str(traj_end)
    ET.SubElement(traj, "step").text = str(traj_step)


def _check_time_range(expr, nt, what):
    pieces = str(expr).split()
    if not pieces:
        raise ValueError(f"{what}: empty time expression")
    for piece in pieces:
        if not re.match(TIME_RANGE, piece):
            raise ValueError(f"{what}: '{piece}' is not a number or a..b")
        lo, _, hi = piece.partition("..")
        if int(lo) > int(hi or lo) or int(hi or lo) >= nt:
            raise ValueError(f"{what}: '{piece}' is not an increasing range below nt = {nt}")


def axis(dense, n_low, time_offset=0):
    """One mode axis (AxisPar). A dense axis is a compact W leg, any other an
    expanded mode index; time_offset is 0 on every base field."""
    return dict(dense=bool(dense), nLow=int(n_low), timeOffset=int(time_offset))


def matrix_par(name, file, dataset, row, col, time_slice_io=True):
    """One <a2aMatrix> entry (MatrixPar). `file` carries the @traj@ token;
    with time_slice_io it is the whole-field stem <x>.h5 and slice t is read
    from <x>.tNNNN.h5. `dataset` is the HDF5 group (the ioname)."""
    el = ET.Element("elem")
    _add_option(el, "name", name)
    _add_option(el, "file", file)
    _add_option(el, "dataset", dataset)
    _add_option(el, "timeSliceIO", bool(time_slice_io))
    for side, ax in (("row", row), ("col", col)):
        a = ET.SubElement(el, side)
        _bool(a, "dense", ax["dense"])
        _add_option(a, "nLow", ax["nLow"])
        _add_option(a, "timeOffset", ax["timeOffset"])
    return el


class _DenseJob:
    """<global> plus the shared <a2aMatrix> list; subclasses add their own
    entry list."""

    def __init__(self, nt, traj_start, traj_end, traj_step=1, output=None):
        self.nt = nt
        self.root = ET.Element("grid")
        glob = ET.SubElement(self.root, "global")
        _traj(glob, traj_start, traj_end, traj_step)
        _add_option(glob, "nt", nt)
        if output is not None:
            _add_option(glob, "output", output)
        self.matrices_el = ET.SubElement(self.root, "a2aMatrix")
        self.matrices = {}

    def add_matrix(self, name, file, dataset, row, col, time_slice_io=True):
        if name in self.matrices:
            raise ValueError(f"duplicate a2aMatrix name '{name}'")
        if time_slice_io and not str(file).endswith(".h5"):
            raise ValueError(f"timeSliceIO file '{file}' must end in .h5; the "
                             f"utility inserts .tNNNN before that suffix")
        for side, ax in (("row", row), ("col", col)):
            if not 0 <= ax["timeOffset"] < self.nt:
                raise ValueError(f"'{name}' {side} timeOffset {ax['timeOffset']} "
                                 f"is not in 0..nt-1")
        self.matrices[name] = dict(file=file, dataset=dataset, row=row, col=col,
                                   time_slice_io=time_slice_io)
        self.matrices_el.append(matrix_par(name, file, dataset, row, col, time_slice_io))
        return name

    def _check_declared(self, name, what):
        if name not in self.matrices:
            raise ValueError(f"{what}: '{name}' is not a declared a2aMatrix")

    def write(self, xml_path):
        self._validate()
        return _write(self.root, xml_path)


class PixPiJob(_DenseJob):
    """An A2APixPi par file: products C[t] = L[t + delta_l] R[t + delta_r],
    one timeSliceIO output per product. The deltas are unsigned with at least
    one zero, so a product is labelled by its unshifted factor. Inputs must be
    base fields: zero time offsets, not products."""

    def __init__(self, nt, traj_start, traj_end, traj_step=1):
        super().__init__(nt, traj_start, traj_end, traj_step)
        self.products_el = ET.SubElement(self.root, "product")
        self.products = []

    def add_product(self, left, right, delta_l, delta_r, output, dataset):
        """Returns the product's a2aMatrix entry as keyword arguments for a
        contractor job's add_matrix: rows are L's rows offset by delta_l, cols
        R's cols offset by delta_r."""
        what = f"product '{output}'"
        self._check_declared(left, what)
        self._check_declared(right, what)
        el = ET.SubElement(self.products_el, "elem")
        _add_option(el, "left", left)
        _add_option(el, "right", right)
        _add_option(el, "deltaL", delta_l)
        _add_option(el, "deltaR", delta_r)
        _add_option(el, "output", output)
        _add_option(el, "dataset", dataset)
        self.products.append((left, right, delta_l, delta_r, output))
        lrow, rcol = self.matrices[left]["row"], self.matrices[right]["col"]
        return dict(file=output, dataset=dataset, time_slice_io=True,
                    row=axis(lrow["dense"], lrow["nLow"], (lrow["timeOffset"] + delta_l) % self.nt),
                    col=axis(rcol["dense"], rcol["nLow"], (rcol["timeOffset"] + delta_r) % self.nt))

    def _validate(self):
        outputs = set()
        for left, right, delta_l, delta_r, output in self.products:
            what = f"product '{output}'"
            if delta_l >= self.nt or delta_r >= self.nt or (delta_l and delta_r):
                raise ValueError(f"{what}: need deltaL, deltaR < nt with at least one zero")
            if not str(output).endswith(".h5"):
                raise ValueError(f"{what}: output must end in .h5")
            if output in outputs:
                raise ValueError(f"{what}: duplicate output")
            outputs.add(output)
            for name in (left, right):
                m = self.matrices[name]
                if m["row"]["timeOffset"] or m["col"]["timeOffset"]:
                    raise ValueError(f"{what}: input '{name}' has a time offset; "
                                     f"inputs must be base fields")


class FlexibleJob(_DenseJob):
    """A FlexibleContractorDense par file.

    A contraction is tr(T_0 ... T_{n-1}) with each term a (name, role, offset):
    a source sits at dt + offset, a sink at tLast + offset, offsets taken mod
    nt. For every dt in `translations` the sink sweeps tLast = dt + s for s in
    min_t..max_t and bins at s. The first term is a source and cyclic
    neighbours have different roles: a fixed-offset pair is a product, built
    with PixPiJob. Without translation_average each dt is its own dataset
    <name>_dt_<dt>; with it, one averaged dataset <name>. One file per
    contraction, <output>/<name>.<traj>.h5."""

    ROLES = ("source", "sink")

    def __init__(self, nt, output, traj_start, traj_end, traj_step=1):
        super().__init__(nt, traj_start, traj_end, traj_step, output)
        self.contractions_el = ET.SubElement(self.root, "contraction")
        self.contractions = {}

    def add_contraction(self, name, terms, translations, min_t, max_t,
                        translation_average):
        if name in self.contractions:
            raise ValueError(f"duplicate contraction name '{name}'")
        terms = [tuple(t) for t in terms]
        el = ET.SubElement(self.contractions_el, "elem")
        _add_option(el, "name", name)
        terms_el = ET.SubElement(el, "terms")
        for term, role, offset in terms:
            t = ET.SubElement(terms_el, "elem")
            _add_option(t, "name", term)
            _add_option(t, "role", role)
            _add_option(t, "offset", offset)
        _add_option(el, "translations", translations)
        _add_option(el, "min_t", min_t)
        _add_option(el, "max_t", max_t)
        _bool(el, "translationAverage", translation_average)
        self.contractions[name] = (terms, translations, min_t, max_t)
        return name

    def _validate(self):
        for name, (terms, translations, min_t, max_t) in self.contractions.items():
            what = f"contraction '{name}'"
            if len(terms) < 2:
                raise ValueError(f"{what}: needs at least two terms")
            for term, role, _ in terms:
                self._check_declared(term, what)
                if role not in self.ROLES:
                    raise ValueError(f"{what}: role '{role}' is not source or sink")
            if terms[0][1] != "source":
                raise ValueError(f"{what}: the first term must be a source")
            for j, (term, role, _) in enumerate(terms):
                nxt = terms[(j + 1) % len(terms)]
                if role == nxt[1]:
                    raise ValueError(f"{what}: neighbours '{term}' and '{nxt[0]}' are "
                                     f"both {role}s; build the pair with PixPiJob")
            if not min_t <= max_t < self.nt:
                raise ValueError(f"{what}: need min_t <= max_t < nt")
            _check_time_range(translations, self.nt, what)


class BubbleJob(_DenseJob):
    """A BubbleContractorDense par file.

    A bubble is B[t] = tr(T_0[t + o_0] ...) for every t in 0..nt-1, with one
    term (a self-trace: the sigma loop, or a two-pion bubble as tr of a
    PixPiJob product) or two (a pair with no product). One file per bubble,
    <output>/<name>.<traj>.h5, holding the single dataset <name>."""

    def __init__(self, nt, output, traj_start, traj_end, traj_step=1):
        super().__init__(nt, traj_start, traj_end, traj_step, output)
        self.bubbles_el = ET.SubElement(self.root, "bubble")
        self.bubbles = {}

    def add_bubble(self, name, terms):
        if name in self.bubbles:
            raise ValueError(f"duplicate bubble name '{name}'")
        terms = [tuple(t) for t in terms]
        el = ET.SubElement(self.bubbles_el, "elem")
        _add_option(el, "name", name)
        terms_el = ET.SubElement(el, "terms")
        for term, offset in terms:
            t = ET.SubElement(terms_el, "elem")
            _add_option(t, "name", term)
            _add_option(t, "offset", offset)
        self.bubbles[name] = terms
        return name

    def _validate(self):
        for name, terms in self.bubbles.items():
            what = f"bubble '{name}'"
            if len(terms) not in (1, 2):
                raise ValueError(f"{what}: needs one or two terms; a longer fixed-"
                                 f"offset chain is a product, build it with PixPiJob")
            for term, _ in terms:
                self._check_declared(term, what)


class K2pipiJob(_DenseJob):
    """A K2pipiContractorDense par file.

    A contraction is C_E(dt, tau) = tr(F(dt) E[dt + tau]) with the fixed chain
    F(dt) = T_0[dt + o_0] ... T_{k-1}[dt + o_{k-1}], terms (name, offset) with
    k >= 1, and E each field of `swept`. tau runs over min_t..max_t, dt over
    `translations`. The fixed fields are loaded whole and every F(dt) is kept;
    each swept field is streamed one slice at a time, read once for all the
    contractions that list it. One file per contraction,
    <output>/<name>.<traj>.h5, holding per swept field E the datasets
    <name>_<E>_dt_<dt>, or <name>_<E> with translation_average."""

    def __init__(self, nt, output, traj_start, traj_end, traj_step=1):
        super().__init__(nt, traj_start, traj_end, traj_step, output)
        self.contractions_el = ET.SubElement(self.root, "contraction")
        self.contractions = {}

    def add_contraction(self, name, terms, swept, translations, min_t, max_t,
                        translation_average):
        if name in self.contractions:
            raise ValueError(f"duplicate contraction name '{name}'")
        terms = [tuple(t) for t in terms]
        swept = list(swept)
        el = ET.SubElement(self.contractions_el, "elem")
        _add_option(el, "name", name)
        terms_el = ET.SubElement(el, "terms")
        for term, offset in terms:
            t = ET.SubElement(terms_el, "elem")
            _add_option(t, "name", term)
            _add_option(t, "offset", offset)
        _add_option(el, "swept", swept)
        _add_option(el, "translations", translations)
        _add_option(el, "min_t", min_t)
        _add_option(el, "max_t", max_t)
        _bool(el, "translationAverage", translation_average)
        self.contractions[name] = (terms, swept, translations, min_t, max_t)
        return name

    def _validate(self):
        for name, (terms, swept, translations, min_t, max_t) in self.contractions.items():
            what = f"contraction '{name}'"
            if not terms:
                raise ValueError(f"{what}: needs at least one fixed term")
            if not swept:
                raise ValueError(f"{what}: needs at least one swept field")
            if len(set(swept)) != len(swept):
                raise ValueError(f"{what}: a swept field is listed twice")
            for term in [t for t, _ in terms] + swept:
                self._check_declared(term, what)
            if not min_t <= max_t < self.nt:
                raise ValueError(f"{what}: need min_t <= max_t < nt")
            _check_time_range(translations, self.nt, what)


def select_axis(low_size, modes_per_hit, n_hit_in, is_v, hits):
    """One mode axis of an A2AMesonFieldSelectHits selection. The output axis
    is the low block, unscaled, then the listed hits' blocks in list order; on
    a V axis those are scaled by n_hit_in/len(hits), which trades the input's
    1/n_hit_in for the output's own. The utility checks
    low_size + n_hit_in*modes_per_hit against the dataset extent."""
    return dict(lowSize=low_size, modesPerHit=modes_per_hit, nHitIn=n_hit_in,
                isV=is_v, hits=list(hits))


class SelectHitsJob:
    """One A2AMesonFieldSelectHits par file: a <global> block and one
    <selection> entry per HDF5 file.

    The utility is serial and single-process, so a par file is the unit of
    parallelism, as it is for the contractor. It refuses to overwrite an
    existing output, so a rerun needs the previous outputs removed first."""

    def __init__(self, traj_start, traj_end, traj_step=1):
        self.root = ET.Element("grid")
        glob = ET.SubElement(self.root, "global")
        _traj(glob, traj_start, traj_end, traj_step)
        self.selection_el = ET.SubElement(self.root, "selection")
        self.n = 0

    def add(self, file, dataset, output, left, right):
        el = ET.SubElement(self.selection_el, "elem")
        _add_option(el, "file", file)
        _add_option(el, "dataset", dataset)
        _add_option(el, "output", output)
        for side, ax in (("left", left), ("right", right)):
            a = ET.SubElement(el, side)
            for key in ("lowSize", "modesPerHit", "nHitIn"):
                _add_option(a, key, ax[key])
            _bool(a, "isV", ax["isV"])
            _add_option(a, "hits", ax["hits"])
        self.n += 1

    def write(self, xml_path):
        return _write(self.root, xml_path)
