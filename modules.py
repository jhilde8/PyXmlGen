"""
One factory function per Hadrons module type used by the job generators.
Each function's keyword arguments match that module's Serializable Par
struct field-for-field (verified against the .hpp sources under
Hadrons/Hadrons/Modules/), so a typo'd or missing argument is a Python
TypeError at generation time instead of silently-wrong XML.
"""
from hadrons_xml import module


def load_binned_a2a_vecs(name, bin_size, filestem, size, multi_file=True):
    return module(name, f"MIO::LoadBinnedA2AVecs{bin_size}",
                  filestem=filestem, multiFile=multi_file, size=size)


def load_combined_a2a_vecs_v(name, low_filestem, n_low, high_stem, high_extensions,
                            high_size, low_bin_size, high_bin_size, n_hit=0):
    # n_hit: 1/nHit hit-average normalization on the high-mode blocks; the
    # factor lives on the V side by convention, so set it (to the number of
    # hits in the job's estimator) only for V loads. 0 or 1 = no rescaling
    # (W loads, single-hit data).
    return module(name, f"MIO::LoadCombinedA2AVecsV{low_bin_size}x{high_bin_size}",
                  lowFilestem=low_filestem, nLow=n_low, highStem=high_stem,
                  highExtensions=high_extensions, highSize=high_size, nHit=n_hit)


def time_diluted_noise(name):
    return module(name, "MNoise::TimeDilutedSpinColorDiagonal")


def save_noise(name, filestem, noise):
    return module(name, "MIO::SaveSpinColorDiagonalNoise",
                  filestem=filestem, noise=noise)


def load_time_diluted_noise(name, file_stems, n_noise_per_file_stem=1):
    # Hit order of the resulting noise object (and so of every dense W array
    # expanded from it) is the file_stems order given here.
    return module(name, "MIO::LoadTimeDilutedSpinColorDiagonalNoise",
                  fileStems=file_stems, nNoisePerFileStem=n_noise_per_file_stem)


def load_combined_a2a_vecs_w(name, low_bin_size, low_filestem, n_low, noise):
    # Dense/combined high-mode W representation: nLow low modes from binned
    # files, then 12 dense fermion fields per hit expanded from `noise`
    # (slot = nLow + hit*12 + sc, sc fastest). No nHit normalization: W stays
    # raw noise.
    return module(name, f"MIO::LoadCombinedA2AVecsW{low_bin_size}",
                  lowFilestem=low_filestem, nLow=n_low, noise=noise)


def a2a_high_mode_v_binned(name, bin_size, noise, action, solver, output,
                            multi_file=True):
    return module(name, f"MSolver::A2AHighModeVBinned{bin_size}",
                  noise=noise, action=action, solver=solver, output=output,
                  multiFile=multi_file)


def a2a_low_mode_coarse_binned(name, n_basis, bin_size, eigen_pack, action,
                                output, schur_convention="", check_interval=0):
    return module(name, f"MUtilities::A2ALowModeCoarseBinned{n_basis}Bin{bin_size}",
                  eigenPack=eigen_pack, action=action, output=output,
                  schurConvention=schur_convention, checkInterval=check_interval)


def a2a_low_mode_binned(name, bin_size, action, eigen_pack, output,
                         multi_file=True):
    return module(name, f"MUtilities::A2ALowModeBinned{bin_size}",
                  action=action, eigenPack=eigen_pack, output=output,
                  multiFile=multi_file)


def load_binned_a2a_vecs_v(name, low_bin_size, high_bin_size, low_filestem,
                            high_file_stems, low_size, high_size, n_hit,
                            multi_file=True):
    return module(name, f"MIO::LoadBinnedA2AVecsV{low_bin_size}_{high_bin_size}",
                  lowFilestem=low_filestem, highFileStems=high_file_stems,
                  multiFile=multi_file, lowSize=low_size, highSize=high_size,
                  nHit=n_hit)


def load_binned_a2a_vecs_w(name, bin_size, filestem, low_size, noise,
                            multi_file=True):
    return module(name, f"MIO::LoadBinnedA2AVecsW{bin_size}",
                  filestem=filestem, multiFile=multi_file, lowSize=low_size,
                  noise=noise)


def vector_pack_ref_slice(name, source, offsets, counts):
    return module(name, "MUtilities::FermionVectorPackRefSlice",
                  source=source, offsets=offsets, counts=counts)


def a2a_covariant_smear(name, a2a_vectors, gauge, alpha, N, orthog_axis=3,
                         output="", multi_file=False):
    return module(name, "MUtilities::A2ACovariantSmear",
                  a2aVectors=a2a_vectors, gauge=gauge, alpha=alpha, N=N,
                  orthog_axis=orthog_axis, output=output, multiFile=multi_file)


def a2a_coarse_grid(name, bin_size, fine, block_size, offsets, output):
    return module(name, f"MUtilities::A2ACoarseGrid{bin_size}",
                  fine=fine, blockSize=block_size, offsets=offsets, output=output)


def a2a_meson_field(name, block, cache_block, left, right, output, gammas, mom,
                     time_slice_io=False):
    # time_slice_io: skip the temporal all-gather and write one file per
    # (momentum, gamma, global timeslice), named "<ioname>.tNNNN.h5", spread
    # over min(nt, nRank) writers instead of nmom*ngamma of them. Worth it
    # for anything with a fat leg; not for the thin kaon fields, where 128x
    # the file count is all Lustre metadata for a sub-second write.
    return module(name, "MContraction::A2AMesonField",
                  block=block, cacheBlock=cache_block, left=left, right=right,
                  output=output, gammas=gammas, mom=mom,
                  timeSliceIO=time_slice_io)

def a2a_new_meson_field(name, block, cache_block, left, right, output, gammas, mom):
    return module(name, "MContraction::A2ANewMesonField",
                  block=block, cacheBlock=cache_block, left=left, right=right,
                  output=output, gammas=gammas, mom=mom)

def a2a_loop(name, left, right):
    # Original loop builder: loop = sum_k outerProduct(left[k], right[k]),
    # plain index-for-index pairing over the whole array, so `right` must be
    # the fully expanded W. Kept untouched as the independent reference for
    # the A2ALoopNew regression -- do not reroute it through A2ALoopNew.
    return module(name, "MContraction::A2ALoop", left=left, right=right)


def a2a_loop_new(name, left, right, n_low, block):
    # Superset of a2a_loop: takes either W representation (deduced from the
    # array sizes and logged), blocks the mode sum so device residency is set
    # by `block` rather than by the mode count, and for dense W compresses
    # each hit's expanded V block to N_SC fields before contracting.
    # n_low must match the low-mode block both loaders were given.
    return module(name, "MContraction::A2ALoopNew",
                  left=left, right=right, nLow=n_low, block=block)


def write_prop(name, prop, file, format="IEEE64BIG"):
    # Raw lexicographic BinaryIO write (no header, no checksum verification);
    # the module appends ".<traj>" to `file`. `format` must match whatever
    # MIO::LoadProp / an offline reader is told to expect.
    return module(name, "MIO::WriteProp", prop=prop, file=file, format=format)


def load_prop(name, file, format="IEEE64BIG"):
    return module(name, "MIO::LoadProp", file=file, format=format)


def a2a_extended_meson_field(name, block, cache_block, types, left, right,
                              output, gammas1, gammas2, loop="",
                              loop_vw1="", loop_vw2="", time_slice_io=False):
    # The quark loop comes in exactly one of two ways -- `loop`, naming a
    # precomputed PropagatorField (the production path), or `loop_vw1`/
    # `loop_vw2`, naming the loop's vector arrays for in-module construction.
    # The latter is expanded-W only and exists as the verification reference;
    # it goes away with the corresponding branch in the module.
    #
    # time_slice_io also shrinks mBuf, which EMF sizes ntOut*N_i*N_j rather
    # than nt*N_i*N_j -- a factor P_t off the per-rank footprint, not just a
    # different file layout.
    if bool(loop) == bool(loop_vw1 or loop_vw2):
        raise ValueError(f"module '{name}': pass either loop= or "
                         f"loop_vw1=/loop_vw2=, not both or neither")
    if not loop and not (loop_vw1 and loop_vw2):
        raise ValueError(f"module '{name}': the vector path needs both "
                         f"loop_vw1 and loop_vw2")
    return module(name, "MContraction::A2AExtendedMesonField",
                  block=block, cacheBlock=cache_block, types=types,
                  left=left, right=right, loop=loop,
                  loop_vw1=loop_vw1, loop_vw2=loop_vw2,
                  output=output, gammas1=gammas1, gammas2=gammas2,
                  timeSliceIO=time_slice_io)


def a2a_chromomagnetic_operator_field(name, block, cache_block, parities,
                                       left, right, gauge, output, if_orthogs,
                                       time_slice_io=False):
    # See a2a_extended_meson_field on time_slice_io: CMOF sizes mBuf the same
    # way, so this is a memory switch as much as an IO one.
    return module(name, "MContraction::A2AChromoMagneticOperatorField",
                  block=block, cacheBlock=cache_block, parities=parities,
                  left=left, right=right, gauge=gauge, output=output,
                  ifOrthogs=if_orthogs, timeSliceIO=time_slice_io)


def load_nersc(name, file):
    return module(name, "MIO::LoadNersc", file=file)


def ape_smear(name, gauge, alpha, N, orthog_axis=3):
    return module(name, "MGauge::APESmear",
                  gauge=gauge, alpha=alpha, N=N, orthog_axis=orthog_axis)
