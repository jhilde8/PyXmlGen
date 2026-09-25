"""
XML construction for the standalone contractor utilities.

This is a different document from hadrons_xml.Job. A Hadrons job is
<grid><parameters/><modules/></grid> and is executed by the scheduler; a
contractor par file is <grid><global/><a2aMatrix/><product/></grid> and is
read straight into three Serializable structs by main(). Nothing is shared
but the option-serialisation helpers, so the two live side by side rather
than one wrapping the other.

One builder covers both binaries. Passing n_hit to ContractorJob and n_low to
add_matrix emits the ContractorDense schema (Hadrons/utilities/
ContractorDense.cpp); leaving both out emits the stock HadronsContractor one
(Contractor.cpp). Mixing them is rejected on write, since a par file carrying
half the dense fields parses on neither binary.

Element order follows the GRID_SERIALIZABLE_CLASS_MEMBERS declaration order in
the .cpp. Grid's XmlReader looks members up by name so order is not load
bearing, but a par file that reads top to bottom like the struct is easier to
check against it.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from hadrons_xml import _add_option, _xml_scalar

# parseTimeRange's own regex (Contractor.cpp): a bare timeslice or an
# inclusive a..b range, no sign. A negative offset is written as nt - k and
# left to TIME_MOD.
TIME_RANGE = r"^([0-9]+|[0-9]+\.\.[0-9]+)$"


def _elem(parent, **fields):
    el = ET.SubElement(parent, "elem") if parent is not None else ET.Element("elem")
    for key, val in fields.items():
        if val is not None:
            _add_option(el, key, val)
    return el


def global_par(nt, disk_vector_dir, output, traj_start, traj_end, traj_step=1,
               n_hit=None):
    """The <global> block. n_hit is the hit count every dense field in the par
    file shares, and setting it selects the ContractorDense schema.

    The two directories have opposite requirements, both of them traps:

      output          MUST already exist. saveCorrelator calls
                      makeFileDir(dir), but makeFileDir creates dirname() of
                      what it is handed -- so passing the output directory
                      creates its PARENT and never the directory itself, and
                      ResultWriter then fails to create a file inside a
                      directory nobody made. Create it in the submission
                      script. Affects Contractor, ContractorDense,
                      FlexibleContractor and BubbleContractor alike, since
                      they share saveCorrelator.

      diskVectorDir   its per-matrix subdirectory, <diskVectorDir>/<name>,
                      must NOT exist. DiskVectorBase hard-errors on a
                      pre-existing one (DiskVector.hpp:274-283) before
                      creating it, deliberately, since it wipes what it owns.
                      Hadrons::mkdir is create_directories, so the parent
                      chain comes for free -- only the leaf must be absent.
                      A crashed run therefore needs it removed before a retry.
    """
    el = ET.Element("global")
    traj = ET.SubElement(el, "trajCounter")
    ET.SubElement(traj, "start").text = str(traj_start)
    ET.SubElement(traj, "end").text = str(traj_end)
    ET.SubElement(traj, "step").text = str(traj_step)
    _add_option(el, "nt", nt)
    if n_hit is not None:
        _add_option(el, "nHit", n_hit)
    _add_option(el, "diskVectorDir", disk_vector_dir)
    _add_option(el, "output", output)
    return el


def a2a_matrix(file, dataset, name, cache_size, n_low=None, time_slice_io=False):
    """One <a2aMatrix> entry. `file` carries the @traj@ token; `dataset` is the
    HDF5 group, which for a meson field is the ioname that A2AMesonField built
    from the gamma and momentum (the dataset inside it is always a2aMatrix).

    n_low is the low-mode count of this field's DENSE (row) axis only. Its
    column axis belongs to the other term of the contraction, and is checked
    against that term's mode space at run time -- so one value per matrix both
    suffices and cross-checks the pairing.

    time_slice_io says the field was written one file per timeslice, in which
    case `file` is still the whole-field path and the contractor inserts
    ".t%04d" before the .h5 the way A2AMesonField's filenameFn does. Both it
    and n_low belong to the ContractorDense schema, so they are emitted
    together or not at all."""
    el = ET.Element("elem")
    _add_option(el, "file", file)
    _add_option(el, "dataset", dataset)
    _add_option(el, "cacheSize", cache_size)
    if n_low is not None:
        _add_option(el, "timeSliceIO", bool(time_slice_io))
        _add_option(el, "nLow", n_low)
    _add_option(el, "name", name)
    return el


def product(terms, times, translations, translation_average):
    """One <product> entry: a trace of the named matrices.

    `terms` is the matrix names in trace order, `times` one time expression per
    term EXCEPT the last -- the last term's time is swept over all nt and sets
    the correlator's abscissa. `translations` shifts every term together."""
    el = ET.Element("elem")
    _add_option(el, "terms", " ".join(terms) if isinstance(terms, (list, tuple)) else terms)
    _add_option(el, "times", list(times))
    _add_option(el, "translations", translations)
    # Not via _add_option: its bool guard lists the Hadrons module options that
    # are genuinely bool, and this one belongs to the contractor's ProductPar.
    ET.SubElement(el, "translationAverage").text = _xml_scalar(translation_average)
    return el


class ContractorJob:
    """One contractor par file: a <global> block plus the matrices and
    products added to it."""

    def __init__(self, nt, disk_vector_dir, output, traj_start, traj_end,
                 traj_step=1, n_hit=None):
        self.nt = nt
        self.n_hit = n_hit
        self.root = ET.Element("grid")
        self.root.append(global_par(nt, disk_vector_dir, output, traj_start,
                                    traj_end, traj_step, n_hit))
        self.matrices_el = ET.SubElement(self.root, "a2aMatrix")
        self.products_el = ET.SubElement(self.root, "product")
        self._matrices = {}
        self._products = []

    def add_matrix(self, file, dataset, name, cache_size, n_low=None,
                   time_slice_io=False):
        if name in self._matrices:
            raise ValueError(f"duplicate a2aMatrix name '{name}'")
        if time_slice_io and not str(file).endswith(".h5"):
            raise ValueError(
                f"timeSliceIO file '{file}' must end in .h5; the contractor "
                f"inserts .t%04d before that suffix")
        self._matrices[name] = n_low
        self.matrices_el.append(a2a_matrix(file, dataset, name, cache_size,
                                           n_low, time_slice_io))
        return name

    def add_product(self, terms, times, translations, translation_average=True):
        terms = list(terms) if isinstance(terms, (list, tuple)) else terms.split()
        times = list(times)
        self._products.append((terms, times, translations))
        self.products_el.append(product(terms, times, translations,
                                        translation_average))

    def _validate(self):
        dense = self.n_hit is not None
        for name, n_low in self._matrices.items():
            if (n_low is None) != (not dense):
                raise ValueError(
                    f"matrix '{name}' {'lacks' if dense else 'carries'} nLow but "
                    f"global {'has' if dense else 'has no'} nHit; a par file is "
                    f"either wholly dense or wholly expanded")
        import re
        for terms, times, translations in self._products:
            # Contractor.cpp:319 -- the last term has no time of its own.
            if len(times) != len(terms) - 1:
                raise ValueError(
                    f"product {terms} has {len(times)} times, expected "
                    f"{len(terms) - 1} (the last term is swept over all nt)")
            if dense and len(terms) != 2:
                raise ValueError(
                    f"product {terms} has {len(terms)} terms; ContractorDense "
                    f"handles 2 until PixPi lands")
            for t in terms:
                if t not in self._matrices:
                    raise ValueError(f"product term '{t}' is not a declared a2aMatrix")
            for expr in list(times) + [translations]:
                for piece in str(expr).split():
                    if not re.match(TIME_RANGE, piece):
                        raise ValueError(
                            f"'{piece}' is not a time expression parseTimeRange "
                            f"accepts (a number or a..b, no sign)")
                    for v in re.findall(r"[0-9]+", piece):
                        if int(v) >= self.nt:
                            raise ValueError(f"time {v} is not below nt = {self.nt}")

    def write(self, xml_path):
        self._validate()
        xml_path = Path(xml_path)
        ET.indent(self.root, space="  ")
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(xml_path, encoding="unicode",
                                        xml_declaration=True)
        return xml_path
