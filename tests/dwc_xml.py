"""
Par files for Masaaki's DWC utilities, the regression references of the
dense contraction suite (contractor_xml):

    ProductJob       A2AMesonFieldProduct     reference for PixPiJob
    FlexibleDWCJob   FlexibleContractorDWC    reference for FlexibleJob
    BubbleDWCJob     BubbleContractorDWC      reference for BubbleJob
    K2pipiDWCJob     K2pipiContractorDWC      reference for K2pipiJob

Test-only: production par files come from contractor_xml. Matrices take the
same axis dicts as contractor_xml (axis(), config.field_axes), translated to
his initDense/timeOffset fields: initDense is nLow on a dense axis and -1
otherwise. His contractors read single files only, through a DiskVector whose
per-matrix directory <diskVectorDir>/<name> must not exist.

Element order follows his struct declarations. Each job also knows the output
stem his utility derives from the terms, so a test can find his result.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contractor_xml import _bool, _traj, _write
from hadrons_xml import _add_option


def _init_dense(ax):
    return ax["nLow"] if ax["dense"] else -1


class _DWCContractorJob:
    """<global> with a DiskVector directory, plus his A2AMatrixPar list."""

    def __init__(self, nt, disk_vector_dir, output, traj_start, traj_end, traj_step=1):
        self.nt = nt
        self.root = ET.Element("grid")
        glob = ET.SubElement(self.root, "global")
        _traj(glob, traj_start, traj_end, traj_step)
        _add_option(glob, "nt", nt)
        _add_option(glob, "diskVectorDir", disk_vector_dir)
        _add_option(glob, "output", output)
        self.matrices_el = ET.SubElement(self.root, "a2aMatrix")
        self.products_el = ET.SubElement(self.root, "product")

    def add_matrix(self, name, file, dataset, row, col):
        el = ET.SubElement(self.matrices_el, "elem")
        _add_option(el, "file", file)
        _add_option(el, "dataset", dataset)
        _add_option(el, "cacheSize", self.nt)
        _add_option(el, "name", name)
        _add_option(el, "initDenseRow", _init_dense(row))
        _add_option(el, "initDenseCol", _init_dense(col))
        _add_option(el, "timeOffsetRow", row["timeOffset"])
        _add_option(el, "timeOffsetCol", col["timeOffset"])

    def write(self, xml_path):
        return _write(self.root, xml_path)


class ProductJob:
    """An A2AMesonFieldProduct par file. One product C[t] = L[t+dL] R[t+dR]
    per entry, single-file output; ifDenseL/R flag the dense side of the
    inner edge (L's col, R's row) and initDense is L's nLow."""

    def __init__(self, traj_start, traj_end, traj_step=1):
        self.root = ET.Element("grid")
        _traj(ET.SubElement(self.root, "global"), traj_start, traj_end, traj_step)
        self.matrices_el = ET.SubElement(self.root, "a2aMatrix")
        self.products_el = ET.SubElement(self.root, "product")
        self.axes = {}

    def add_matrix(self, name, file, group, row, col):
        el = ET.SubElement(self.matrices_el, "elem")
        _add_option(el, "name", name)
        _add_option(el, "file", file)
        _add_option(el, "group", group)
        self.axes[name] = (row, col)

    def add_product(self, left, right, delta_l, delta_r, output, group):
        el = ET.SubElement(self.products_el, "elem")
        _add_option(el, "mfL", left)
        _add_option(el, "mfR", right)
        _add_option(el, "deltaL", delta_l)
        _add_option(el, "deltaR", delta_r)
        _bool(el, "ifDenseL", self.axes[left][1]["dense"])
        _bool(el, "ifDenseR", self.axes[right][0]["dense"])
        _add_option(el, "initDense", self.axes[left][0]["nLow"])
        _add_option(el, "output", output)
        _add_option(el, "outputGroup", group)

    def write(self, xml_path):
        return _write(self.root, xml_path)


class FlexibleDWCJob(_DWCContractorJob):
    """A FlexibleContractorDWC par file. Terms are (name, sink, offset) with
    sink his tdps flag; the output stem is name_{x|y}offset per term, x for a
    sink, joined by '_'."""

    def add_contraction(self, terms, translations, min_t, max_t, translation_average):
        el = ET.SubElement(self.products_el, "elem")
        terms_el = ET.SubElement(el, "terms")
        for name, sink, offset in terms:
            t = ET.SubElement(terms_el, "elem")
            _add_option(t, "term", name)
            _bool(t, "tdps", sink)
            _add_option(t, "time", offset)
        _add_option(el, "translations", translations)
        _add_option(el, "min_t", min_t)
        _add_option(el, "max_t", max_t)
        _bool(el, "translationAverage", translation_average)
        return self.stem(terms)

    @staticmethod
    def stem(terms):
        return "_".join(f"{n}_{'x' if s else 'y'}{o}" for n, s, o in terms)


class BubbleDWCJob(_DWCContractorJob):
    """A BubbleContractorDWC par file. Term 0 sits at dt and times[j-1] is the
    offset of term j from it; the output stem is term0_times0_..._termLast.
    Products with equal stems are written once, since he refuses duplicates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stems = set()

    def add_bubble(self, names, times):
        stem = self.stem(names, times)
        if stem not in self.stems:
            self.stems.add(stem)
            el = ET.SubElement(self.products_el, "elem")
            _add_option(el, "terms", " ".join(names))
            _add_option(el, "times", list(times))
        return stem

    @staticmethod
    def stem(names, times):
        return "".join(f"{n}_{t}_" for n, t in zip(names, times)) + names[-1]


class K2pipiDWCJob(_DWCContractorJob):
    """A K2pipiContractorDWC par file. Terms are (name, offset), the fixed
    chain at dt + offset; lastMats are his swept fields, each also listed once
    in the top-level <a2aMatrixLast>, which write() fills. The output stem is
    name_offset per term joined by '_'; datasets are <stem>_<last>_dt_<dt>, or
    <stem>_<last> averaged. He refuses two products with one stem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_el = ET.Element("a2aMatrixLast")
        self.root.insert(list(self.root).index(self.products_el), self.last_el)
        self.last = []
        self.stems = set()

    def add_contraction(self, terms, last_mats, translations, min_t, max_t,
                        translation_average):
        stem = self.stem(terms)
        if stem in self.stems:
            raise ValueError(f"duplicate fixed chain '{stem}'")
        self.stems.add(stem)
        el = ET.SubElement(self.products_el, "elem")
        _add_option(el, "terms", " ".join(name for name, _ in terms))
        _add_option(el, "times", [offset for _, offset in terms])
        _add_option(el, "lastMats", list(last_mats))
        _add_option(el, "translations", translations)
        _add_option(el, "min_t", min_t)
        _add_option(el, "max_t", max_t)
        _bool(el, "translationAverage", translation_average)
        self.last += [m for m in last_mats if m not in self.last]
        return stem

    def write(self, xml_path):
        for name in self.last:
            ET.SubElement(self.last_el, "elem").text = name
        return super().write(xml_path)

    @staticmethod
    def stem(terms):
        return "_".join(f"{name}_{offset}" for name, offset in terms)
