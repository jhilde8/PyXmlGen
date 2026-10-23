"""
Site-specific constants for the A2A vector layout (N_HIT hits per
configuration) and physics parameters. Paths are checked against the Frontier
tree when the generated XML is reviewed there before submission.
"""
from pathlib import Path

N_HIT = 8

# AMA correction hit: one sloppy hit beyond the production set, plus an exact
# solve on the same noise. Taking the correction from its own hit keeps it
# uncorrelated with the sloppy observable it corrects, and one exact hit on
# every configuration reaches the exact-to-sloppy ratio the 48I campaign gets
# from exact solves on 10% of its configurations. Both carry hit index N_HIT --
# same noise stem, different vector tree (VW_SLOPPY / VW_EXACT).
AMA_HIT = N_HIT

# Flavours solved exact on every hit, the AMA hit included, so they have one
# vector set and no sloppy/exact distinction. Charm enters only through its
# loop's GIM subtraction against the light loop, so with one charm loop in both
# AMA terms it cancels from the correction; the exact solve costs ~18% more per
# source than sloppy, roughly what dropping the AMA hit's duplicate saves.
EXACT_ONLY_FLAVORS = {"c"}

# --- vector file layout -----------------------------------------------
# Base directory holding the vw/ subdirectory of A2A vector files.
VW_BASE = "/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/vw"

# Low modes: one set (light only), one file per vector, filestem
# "<LOW_VW>/lo_v" / "lo_w" (module appends ".<traj>/elemN.bin").
N_LOW = 2000
LOW_VW = f"{VW_BASE}/low"

# High modes: per flavor, per hit, one file per vector. The tree is split by
# solve accuracy: every sloppy hit under VW_SLOPPY as "<flavor><hit>_v" /
# "<flavor><hit>_w", and the AMA hit's exact solve under VW_EXACT as
# "<flavor><hit>_exact". An EXACT_ONLY_FLAVORS flavour has every hit under
# VW_EXACT as "<flavor><hit>_v", whatever accuracy is asked for. The loaders
# take a stem and one extension per hit, which is what high_stem() and
# high_extension() below produce.
N_HIGH = 1536
VW_SLOPPY = f"{VW_BASE}/sloppy"
VW_EXACT = f"{VW_BASE}/exact"

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


def leg_size(flavor, vw, n_hit):
    """Mode count of the combined (flavor, vw) array at n_hit hits: the low
    modes if the flavor has any, then N_HIGH expanded V or N_SC dense W per
    hit. At 8 hits: l_v 14288, l_w 2096, s_v/c_v 12288, s_w/c_w 96."""
    low = N_LOW if FLAVOR_HAS_LOW[flavor] else 0
    high = N_HIGH if vw == "v" else N_SC
    return low + high * n_hit


# (left leg, right leg) of each meson-field kind, a leg being (flavour, "v" or
# "w"). The left leg is the field's row axis. emf covers all three loops and
# cmf both gauge links: the loop and the links do not touch the mode axes.
FIELD_LEGS = {
    "pi":    (("l", "w"), ("l", "v")),
    "ll":    (("l", "w"), ("l", "v")),
    "sl":    (("s", "w"), ("l", "v")),
    "ls":    (("l", "w"), ("s", "v")),
    "ls_ww": (("l", "w"), ("s", "w")),
    "emf":   (("s", "v"), ("l", "v")),
    "cmf":   (("s", "v"), ("l", "v")),
    "mix":   (("s", "v"), ("l", "v")),
}


def low_filestem(flavor, vw):
    assert FLAVOR_HAS_LOW[flavor]
    return f"{LOW_VW}/lo_{vw}"


def high_stem(flavor, exact=False):
    """Directory the V/W loaders prepend to each hit's extension."""
    if exact or flavor in EXACT_ONLY_FLAVORS:
        return f"{VW_EXACT}/"
    return f"{VW_SLOPPY}/"


def high_extension(flavor, hit, vw="v", exact=False):
    """One hit's high-mode block under high_stem(flavor, exact). The exact
    solve is a V only: W is the raw noise, which the sloppy and exact sets
    share. An EXACT_ONLY_FLAVORS flavour has one V per hit, so `exact` does not
    change its name."""
    if flavor in EXACT_ONLY_FLAVORS:
        if vw != "v":
            raise ValueError(
                f"no W files for '{flavor}{hit}': W is the raw noise")
        return f"{flavor}{hit}_v"
    if exact:
        if vw != "v":
            raise ValueError(
                f"no exact W files for '{flavor}{hit}': W is the shared noise")
        return f"{flavor}{hit}_exact"
    return f"{flavor}{hit}_{vw}"


# Noise directory suffix per flavour. Charm reuses the light sources: its loop
# enters only through the GIM subtraction against the light loop.
NOISE_FLAVOR = {"l": "ud", "s": "s", "c": "ud"}


def noise_filestem(flavor, hit):
    return f"{NOISE_BASE}_{NOISE_FLAVOR[flavor]}/hit00{hit}"


# Loop propagators and sparsened vectors, beside VW_BASE on Lustre. Both are
# written by collective single-file writes (MIO::WriteProp, A2ACoarseGrid),
# which node-local NVMe cannot serve. The charm loop job writes into LOOP_ROOT
# and the contraction job loads from it, so the path lives here once.
LOOP_ROOT = str(Path(VW_BASE).parent / "a2aloop")
SPARSE_ROOT = str(Path(VW_BASE).parent / "sparse")


# --- CG: high-mode V solves (production/gen_cg.py) ------------------------
# Working directory of the CG jobs on Frontier, and where their XML and
# schedule files live (cf. XML_DIR for the meson-field jobs).
CG_DIR = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/CG"
CG_XML_DIR = f"{CG_DIR}/xml"
CG_GRAPH = f"{CG_DIR}/graph.gv"


def cg_schedule_file(run_id, flavor):
    return f"{CG_XML_DIR}/CG{flavor}/schedule.{run_id}.txt"


CG_LS = 12
CG_BOUNDARY = "1 1 1 1"
CG_TWIST = "0 0 0 0"

# Actions, from the Frontier/CG/*.tmp.xml examples (2026-10-14). Light is
# Mobius; strange and charm are scaled Shamir (ScaledDWF, scale = b + c = 2).
CG_ACTIONS = {
    "l": dict(kind="mobius", mass=0.000678, M5=1.8, b=1.5, c=0.5),
    "s": dict(kind="scaled", mass=0.02661, M5=1.8, scale=2),
    "c": dict(kind="scaled", mass=0.3326, M5=1.8, scale=2),
}

# Light deflation: the coarse eigenpack, and the BLAS guesser built from it.
CG_EIGENPACK = dict(
    filestem="/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/converted/vec",
    size_fine=200, size_coarse=2000, block_size="4 4 4 4 12")
CG_DEFLATION_SIZE = 2000

# Solver settings per (flavour, accuracy), from the same examples. Light runs
# the batched split-grid solver and so has maxPatchup and updateResidual;
# strange and charm run the single-RHS solver, which has neither. Light
# deflation: the sloppy solve takes the outer guess only, the exact solve both.
# Light sloppy is 400 inner iterations (decided 2026-10-14; the example still
# has 450) -- fixed across configurations, see plans/CG section 13.1.
CG_SOLVERS = {
    ("l", "sloppy"): dict(max_inner=400, max_outer=1, max_patchup=1,
                          residual=1e-8, update_residual=False, if_cgd=False,
                          inner_guess=False),
    ("l", "exact"):  dict(max_inner=400, max_outer=20, max_patchup=100,
                          residual=1.55e-9, update_residual=True, if_cgd=True,
                          inner_guess=True),
    ("s", "sloppy"): dict(max_inner=400, max_outer=1, residual=1e-8,
                          if_cgd=False),
    ("s", "exact"):  dict(max_inner=500, max_outer=10, residual=1.65e-9,
                          if_cgd=True),
    ("c", "exact"):  dict(max_inner=120, max_outer=10, residual=5e-9,
                          if_cgd=True),
}

# Sources per solver call. Light: one per split-grid partition (64 nodes,
# --batched-solver-split 2.2.2.4 -> 16 partitions). Strange and charm are
# unbatched (A2AHighModeV).
CG_BATCH_SIZE = {"l": 16, "s": 0, "c": 0}
CG_NODES_PER_SRUN = {"l": 64, "s": 4, "c": 4}

# Per-configuration layout: one (n_sloppy, n_exact) pair per srun. The sloppy
# stream is the hits in CG_SLOPPY_HIT_ORDER, the exact stream those in
# CG_EXACT_HIT_ORDER; each srun takes the next n vectors of each, cut into
# per-hit ranges. Balanced on total time per source (CG + write) --
# plans/CG section 13. Timings per source on 4 nodes, write 3.0 s/vector
# (2026-10-16): strange sloppy 29.95 s, exact 71.98 s; charm exact 17.14 s.
CG_LAYOUT = {
    # 4-node sruns, exact only: 27 x 512 = 108 nodes, 2.86 h.
    "c": [(0, 512)] * 27,
    # 4-node sruns: 32 sloppy (3.95 h) + 8 exact (4.00 h) = 160 nodes.
    "s": [(432, 0)] * 32 + [(0, 192)] * 8,
    # 64-node sruns: the exact hit plus 2496 sloppy, then 5664 sloppy twice.
    # Fixed-tolerance batch of 16 (2026-10-16): sloppy 31.8 s, exact 71.5 s
    # (compute + noise), plus 16 writes at 0.21-1.24 s/vector. 2496 balances
    # at 0.37 s/vector, 3.7 h per srun.
    "l": [(2496, 1536), (5664, 0), (5664, 0)],
}

# Light puts the AMA sloppy hit first so that it lands in the same srun as its
# exact solve: one noise load, freed after both.
CG_SLOPPY_HIT_ORDER = {
    "l": [AMA_HIT] + list(range(N_HIT)),
    "s": list(range(N_HIT)) + [AMA_HIT],
    "c": [],
}
CG_EXACT_HIT_ORDER = {
    "l": [AMA_HIT],
    "s": [AMA_HIT],
    "c": list(range(N_HIT)) + [AMA_HIT],
}

# Seconds per solver call, CG plus writes: one batch of CG_BATCH_SIZE sources
# for a batched flavour, one source otherwise. Same measurements as CG_LAYOUT;
# light writes are taken at the 0.37 s/vector its layout balances at. Only
# the ratios matter to gen_cg_repack.py's balance; the absolute values and
# CG_STARTUP only set the walltime it prints.
CG_COST = {
    ("l", "sloppy"): 31.8 + 16 * 0.37,
    ("l", "exact"):  71.5 + 16 * 0.37,
    ("s", "sloppy"): 29.95 + 3.0,
    ("s", "exact"):  71.98 + 3.0,
    ("c", "exact"):  17.14 + 3.0,
}

# Seconds from srun start to the first solve. Light is the eigenpack load
# (~950 s in the 2026-10-19 logs); strange and charm are not measured yet.
CG_STARTUP = {"l": 950.0, "s": 0.0, "c": 0.0}

# Size of one complete V file (elem<i>.bin): a double-precision 64^3 x 128
# spin-colour field, 6442450944 bytes, plus the SciDAC record header.
V_ELEM_BYTES = 6442453432


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

# --- smearing ladder ------------------------------------------------------
# (tag, alpha, N) rungs, applied in order. Each rung smears the PREVIOUS rung's
# arrays, so its alpha and N are an increment, while its tag names the
# cumulative width and total iteration count reached there -- w4p1_n45 is got to
# by an increment of (3.56, 34) on top of w2p03_n11. A2ACovariantSmear moves its
# source, so a rung's arrays are gone once the next rung has run and the peak
# stays one array deep however long the ladder is.
#
# Effective width: the module applies (1 + coeff*Laplacian)^N with
# coeff = alpha^2/(4N), whose Gaussian limit is exp(-N*coeff*p^2), so composing
# rungs adds alpha^2 and the width at a rung is the quadrature sum of the
# increments below it -- see smear_effective_alpha. That holds whether or not the
# rungs share coeff.
#
# Tuned on 48I as (alpha=3, N=24) wide with (1.5, 6) intermediate, both at
# coeff=0.09375. Scaled to 64I by alpha ~ a^-1 (1.73 -> 2.36 GeV) and N ~ a^-2,
# so a 4.1 wide target is reached as 2.03 then 3.56, effective 4.098. The width
# is a tuning choice for excited-state contamination and statistics, so hitting
# 4.1 exactly carries no meaning -- but every job must walk the SAME ladder, or
# an AMA correction differences two different smearings.
SMEAR_LADDER = [
    ("w2p03_n11", 2.03, 11),
    ("w4p1_n45",  3.56, 34),
]
ORTHOG_AXIS = 3


def smear_effective_alpha(tag=None):
    """Effective Gaussian width at ladder rung `tag` (the last rung if None),
    as the quadrature sum of the increments up to and including it."""
    total = 0.0
    for rung_tag, alpha, _ in SMEAR_LADDER:
        total += alpha * alpha
        if rung_tag == tag:
            return total ** 0.5
    if tag is None:
        return total ** 0.5
    known = ", ".join(r[0] for r in SMEAR_LADDER)
    raise ValueError(f"no smearing rung '{tag}' in SMEAR_LADDER (have {known})")

# --- block rules -----------------------------------------------------------
# leftBlock / rightBlock: modes per GEMM side for every A2AMesonField,
# A2AExtendedMesonField and A2AChromoMagneticOperatorField. 256 over 128 was
# measured on 32^3x64 at the 256- and 512-node local volumes: the spatial
# reduce is unchanged (its wire bytes do not depend on block), while the
# per-block costs -- file open/close, Pack, and at the 512-node local volume
# the GEMM -- all drop.
#
# A left block at least as large as the left leg packs that leg once for the
# whole module; generators may override LEFT_BLOCK per field for that. The
# HDF5 chunk is the smaller block among the sides actually split, so keep the
# larger a multiple of it.
LEFT_BLOCK = 256
RIGHT_BLOCK = 256

# --- gauge (for CMO / EMF-adjacent smearing) -------------------------------
# APE_ALPHA is a dimensionless mixing weight between the link and the staple
# average, so it does not scale with the lattice spacing. APE_N sets a smoothing
# radius that grows as sqrt(N) in lattice units, so it scales as a^-2: 48I's 25
# becomes 46 on 64I, taken as 45 to match the covariant smearing's N.
GAUGE_FILE = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/configs/ckpoint_lat"
APE_ALPHA = 0.615384615
APE_N = 45

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


# --- contraction stage (Riker) ---------------------------------------------
# Where the meson fields come to rest after the drain off Frontier's node-local
# NVMe. A2AMesonField writes "<output>.<traj>/<ioname>.h5", or with
# timeSliceIO "<output>.<traj>/<ioname>.t%04d.h5", so a field lives at
# <MF_BASE>/<stem>.<traj>/ and the contractor gets the whole-field path with
# @traj@ still in it.
MF_BASE = "/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/mf"

# The AMA correction hit: one extra sloppy hit plus its exact solve. Same
# layout, different tree, and the fields there are 1-hit -- so a par file
# contracting them needs nHit = 1, not N_HIT. See [[project_ama_correction_hit]].
MF_AMA_BASE = "/lustre/orion/phy157/proj-shared/phy157_dwf/jhilde/main_64I/mf_ama"

# Correlators, and the contractor's own scratch. diskVectorDir is only written
# to when a field spills its cache (DiskVectorBase::evict fires at
# index.size() >= cacheSize), so with cacheSize = N_T it stays empty -- but the
# directory is still created, and it must be per-process, since the contractor
# refuses a pre-existing <diskVectorDir>/<matrix name>.
CONTRACTION_ROOT = "/lustre/orion/phy157/world-shared/jhilde/k2pipipbc/main_64I/A2AContractions"

