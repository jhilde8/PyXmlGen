"""
Site-specific constants for the 4-hit A2A vector layout and physics
parameters. EDIT THE PATHS BELOW to match the real Frontier directory --
they're placeholders copied from Frontier/MF/par.postCG.xml's structure,
not verified locations.
"""
from pathlib import Path

N_HIT = 4

# --- vector file layout -----------------------------------------------
# Base directory holding the vw/ subdirectory of A2A vector files.
VW_BASE = "/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/vw"

# Low modes: one set (light only), 10 files x 200 vectors, filestem
# "<VW_BASE>/l_lo_v" / "l_lo_w" (module appends ".<traj>/elemN.bin").
N_LOW = 2000
LOW_BIN_SIZE = 200
LOW_VW = "/lustre/orion/phy157/scratch/jhilde/64I/vw_lo"

# High modes: per flavor, per hit, 12 files x 128 vectors, filestem
# "<VW_BASE>/<flavor><hit>_v" / "<flavor><hit>_w".
N_HIGH = 1536
HIGH_BIN_SIZE = 128

# Raw noise (dense W path): one file per flavor per hit holding the single
# ComplexField that generates that hit's sources
# (MIO::SaveSpinColorDiagonalNoise / LoadTimeDilutedSpinColorDiagonalNoise).
# The dense W array replaces the N_HIGH expanded fields with N_SC = 12
# combined fields per hit (MIO::LoadCombinedNoise).
NOISE_BASE = "/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/noise"
N_NOISE_PER_STEM = 1
N_SC = 12

# light combined (low+high) = 3536; strange/charm = high only = 1536.
FLAVOR_HAS_LOW = {"l": True, "s": False, "c": False}


def low_filestem(flavor, vw):
    assert FLAVOR_HAS_LOW[flavor]
    return f"{LOW_VW}/{flavor}_lo_{vw}"


def high_filestem(flavor, hit, vw):
    return f"{VW_BASE}/{flavor}{hit}_{vw}"


def noise_filestem(flavor, hit):
    return f"{NOISE_BASE}_{flavor}/hit00{hit}"


# --- lattice geometry -----------------------------------------------------
# Needed by offline readers of raw propagator dumps (Frontier/diff_loop_prop.py),
# which have no Grid to ask. N_T is also N_HIGH / N_SC by construction, since
# the high modes are time-diluted across the full temporal extent.
LATT_SIZE = [64, 64, 64, 128]
N_T = N_HIGH // N_SC

# Byte order for MIO::WriteProp / MIO::LoadProp. These carry no header, so the
# writer and every reader have to be told the same thing.
PROP_IO_FORMAT = "IEEE64BIG"

# --- gamma structures ---------------------------------------------------
GAMMA5 = "Gamma5"
IDENTITY = "Identity"
G5_IDENT = "Identity Gamma5"
EMF_GAMMA_FAMILIES = "GammaMU GammaMUGamma5 Identity Gamma5 SigmaMUNU"    #no tensor structures needed in LEFT/WET BSM basis

# --- momenta --------------------------------------------------------------
ZERO_MOM = [[0, 0, 0]]

# Zero plus all permutations of a single +-1 unit along one axis.
ONE_UNIT_MOM = [
    [0, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
]

# Momentum used by the kaon and sigma meson fields. Swap back to ZERO_MOM
# here if needed -- nothing in gen_kaon.py / gen_pion_sigma.py hardcodes
# either choice, they just reference these names.
KAON_MOM = ONE_UNIT_MOM
SIGMA_MOM = ONE_UNIT_MOM

PION_MOM = [
    [0, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
    [1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0],
    [0, 1, -1], [0, 1, 1], [0, -1, -1], [0, -1, 1],
    [1, 0, -1], [1, 0, 1], [-1, 0, -1], [-1, 0, 1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
]

# --- smearing width -----------------------------------------------------
# (tag, alpha, N) -- tag is used in module/output names. Single width,
# chosen to match the collaboration's 48I alpha=3 width in physical units
# (48I: a^-1=1.73 GeV -> 0.3421 fm; 64I: a^-1=2.36 GeV -> alpha=4.0907
# lattice units, rounded to 4.1) with N picked to hold the same smearing
# coefficient coeff=alpha^2/(4N) as 48I's (alpha=3, N=24) -> coeff=0.09375.
SMEAR_WIDTHS = [
    ("w4p1_n45", 4.1, 45),
]
ORTHOG_AXIS = 3

# --- block / cacheBlock rules ---------------------------------------------
# block: 128 for anything with a strange leg, 221 for light-light.
BLOCK_STRANGE_LEG = 128
BLOCK_LIGHT_LIGHT = 221

# cacheBlock tiles the SumRing reduction. Since the A2ASpatialSum rework the
# GPU path is fastest with no tiling at all -- one tile spanning the whole
# block -- so GPU jobs pass cacheBlock = block and none of the constants below
# are referenced by the generators. They are the CPU tile sizes, kept because
# the same generators serve both builds: on a CPU build the tile is for cache
# locality and wants to be small. Pointing a generator back at CPU means
# putting one of these in its cacheBlock argument by hand.
CB_CPU_MF = 16          # A2AMesonField (kaon, sigma, pion, mix)
CB_CPU_LL = 17          # pion and sigma, integer divisor of 221
CB_CPU_EMF_CMOF = 32    # A2AExtendedMesonField, A2AChromoMagneticOperatorField

# --- gauge (for CMO / EMF-adjacent smearing) -------------------------------
GAUGE_FILE = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/configs/ckpoint_lat"
APE_ALPHA = 0.615384615
APE_N = 25

# --- CMO operator parameters (unchanged from Frontier/MF/par.postCG.xml) --
CMO_PARITIES = "0 1"
CMO_IF_ORTHOGS = "0 1"

# --- coarse-grid sparsening (kaon job, hit (0,0) only -- see project notes) -
COARSE_BLOCK_SIZE = [4, 4, 4, 1]
COARSE_OFFSETS = [0, 0, 0, 0]

# --- output ----------------------------------------------------------------
# Anchored to this file's directory, not the caller's cwd, so `python3
# gen_kaon.py` writes to the same place regardless of where it's invoked from.
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# --- Frontier job file locations -------------------------------------------
# Where every XML and schedule file lives on Frontier. Hadrons resolves
# scheduleFile against the job's cwd, and the submission script cds to the
# Lustre working directory, so a relative "./xml/schedule.*.txt" only works
# because of that choice. Writing it absolute removes the dependency.
XML_DIR = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/MF/xml"

# Placeholder for the meson field output directory. Hadrons resolves a
# relative <output> against the job's cwd, which is on Lustre, so a job meant
# to write to the node-local NVMe has to be given an absolute path -- and that
# path is only known at runtime. Generators emit "<TMP_OUTPUT>/<name>" and the
# submission script substitutes the full destination directory, the same way
# it already substitutes TRAJ_START/TRAJ_END. Use a | delimiter, since the
# replacement is a path:
#
#   sed "s|TMP_OUTPUT|$LOCAL_SCRATCH/mf_bench|g" par.in.xml > par.out.xml
#
# The module appends ".<traj>" and creates the directory itself (Hadrons::mkdir
# is create_directories, so the parent chain comes with it), which is why the
# substituted value is the output directory rather than a file stem.
TMP_OUTPUT = "TMP_OUTPUT"

GRAPH = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/MF/graph.gv"

def schedule_file(run_id):
    """Absolute path Hadrons should read this job's schedule from at runtime."""
    return f"{XML_DIR}/schedule.{run_id}.txt"

