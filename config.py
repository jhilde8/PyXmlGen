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
# TODO: confirm this against the real 4-hit storage location.
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
# TODO: confirm this against the real noise storage location.
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


# --- gamma structures ---------------------------------------------------
GAMMA5 = "Gamma5"
IDENTITY = "Identity"
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

CACHE_BLOCK_MF = 16          # A2AMesonField (kaon, sigma, pion, mix)
CACHE_BLOCK_LL = 17          # pion and sigma cache block, integer divisor of 221
CACHE_BLOCK_EMF_CMOF = 32    # A2AExtendedMesonField, A2AChromoMagneticOperatorField

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
