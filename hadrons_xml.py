"""
Core XML construction for generating Hadrons job files.

A Job wraps one <grid>...</grid> document: the <parameters> block plus a
<modules> list built up by repeated calls to Job.add(). Job.write() emits
both the XML and a companion schedule.txt listing module names in the same
order they were added -- Hadrons frees an environment object once nothing
later in the schedule references it, so insertion order here is also the
memory-management order on the cluster.
"""
import xml.etree.ElementTree as ET
from pathlib import Path


def module(name, type_, **options):
    """Build a <module> element. Keyword args become <options> children:
    lists become repeated <elem> children (for things like <mom>), anything
    else becomes the element's text via str()."""
    m = ET.Element("module")
    id_ = ET.SubElement(m, "id")
    ET.SubElement(id_, "name").text = name
    ET.SubElement(id_, "type").text = type_
    opts = ET.SubElement(m, "options")
    for key, val in options.items():
        _add_option(opts, key, val)
    return m


def _xml_scalar(val):
    # Grid reads bools with std::boolalpha, which expects lowercase
    # true/false -- Python's str(bool) gives "True"/"False" and would
    # silently fail to parse as the intended value.
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


# Option names that really are declared bool in a Hadrons Par struct. A bool
# reaching any other option is a caller mistake, and a silent one: it
# serializes to a well-formed "true"/"false" that only fails when Hadrons
# parses the job on the cluster, reporting "numerical conversion failure on
# 'false' (typeid: j)" against whichever unsigned int field it landed on.
BOOL_OPTIONS = {"multiFile"}


def _add_option(parent, key, val):
    if isinstance(val, bool) and key not in BOOL_OPTIONS:
        raise TypeError(
            f"option '{key}' got the bool {val!r}; only {sorted(BOOL_OPTIONS)} "
            f"are boolean in Hadrons. Pass an int (0/1) if that is what you meant")
    el = ET.SubElement(parent, key)
    if isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, (list, tuple)):
                ET.SubElement(el, "elem").text = " ".join(_xml_scalar(x) for x in item)
            else:
                ET.SubElement(el, "elem").text = _xml_scalar(item)
    else:
        el.text = _xml_scalar(val)


class Job:
    """One Hadrons run: a <grid> tree plus a matching schedule file."""

    def __init__(self, run_id, traj_start="TRAJ_START", traj_end="TRAJ_END",
                 traj_step=10, schedule_file=None, graph_file="graph.gv"):
        self.run_id = run_id
        self.root = ET.Element("grid")
        params = ET.SubElement(self.root, "parameters")
        ET.SubElement(params, "runId").text = run_id
        traj = ET.SubElement(params, "trajCounter")
        ET.SubElement(traj, "start").text = str(traj_start)
        ET.SubElement(traj, "end").text = str(traj_end)
        ET.SubElement(traj, "step").text = str(traj_step)
        genetic = ET.SubElement(params, "genetic")
        ET.SubElement(genetic, "popSize").text = "20"
        ET.SubElement(genetic, "maxGen").text = "1000"
        ET.SubElement(genetic, "maxCstGen").text = "100"
        ET.SubElement(genetic, "mutationRate").text = "0.1"
        ET.SubElement(params, "scheduleFile").text = schedule_file or f"./xml/schedule.{run_id}.txt"
        # We supply a schedule file above, so the genetic scheduler never
        # actually runs -- but leaving schedulerType unset defaults to
        # "genetic" anyway, which makes printSchedule() run a full memory
        # profiling pass (real setup() calls, no garbage collection within
        # one module's dependency closure) just to print a byte count. That
        # pass can OOM on large jobs even though the real run never would.
        scheduler = ET.SubElement(params, "scheduler")
        ET.SubElement(scheduler, "schedulerType").text = "naive"
        ET.SubElement(params, "graphFile").text = graph_file
        ET.SubElement(params, "parallelWriteMaxRetry").text = "1"
        # We hand-order modules == schedule order below, so Hadrons should
        # use our schedule file rather than compute (and overwrite) its own.
        ET.SubElement(params, "saveSchedule").text = "false"

        self.modules_el = ET.SubElement(self.root, "modules")
        self._names = []

    def add(self, module_el):
        name = module_el.find("id/name").text
        if name in self._names:
            raise ValueError(f"duplicate module name '{name}' in job '{self.run_id}'")
        self._names.append(name)
        self.modules_el.append(module_el)
        return name

    def has(self, name):
        return name in self._names

    def write(self, xml_path, schedule_path=None):
        xml_path = Path(xml_path)
        schedule_path = Path(schedule_path) if schedule_path else xml_path.with_suffix(".sched.txt")

        ET.indent(self.root, space="  ")
        tree = ET.ElementTree(self.root)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(xml_path, encoding="unicode", xml_declaration=True)

        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        # TextWriter's vector<string> format: element count first, then one
        # element per line (Grid/serialisation/TextIO.h, writeDefault).
        lines = [str(len(self._names))] + self._names
        schedule_path.write_text("\n".join(lines) + "\n")
