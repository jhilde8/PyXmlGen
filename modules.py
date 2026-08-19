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


def load_combined_a2a_vecs(name, low_filestem, n_low, high_stem, high_extensions,
                            n_high_each, low_bin_size, high_bin_size):
    return module(name, f"MIO::LoadCombinedA2AVecs{low_bin_size}x{high_bin_size}",
                  lowFilestem=low_filestem, nLow=n_low, highStem=high_stem,
                  highExtensions=high_extensions, nHighEach=n_high_each)


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


def a2a_meson_field(name, block, cache_block, left, right, output, gammas, mom):
    return module(name, "MContraction::A2AMesonField",
                  block=block, cacheBlock=cache_block, left=left, right=right,
                  output=output, gammas=gammas, mom=mom)

def a2a_new_meson_field(name, block, cache_block, left, right, output, gammas, mom):
    return module(name, "MContraction::A2ANewMesonField",
                  block=block, cacheBlock=cache_block, left=left, right=right,
                  output=output, gammas=gammas, mom=mom)

def a2a_extended_meson_field(name, block, cache_block, types, left, right,
                              loop_vw1, loop_vw2, output, gammas1, gammas2):
    return module(name, "MContraction::A2AExtendedMesonField",
                  block=block, cacheBlock=cache_block, types=types,
                  left=left, right=right, loop_vw1=loop_vw1, loop_vw2=loop_vw2,
                  output=output, gammas1=gammas1, gammas2=gammas2)


def a2a_chromomagnetic_operator_field(name, block, cache_block, parities,
                                       left, right, gauge, output, if_orthogs):
    return module(name, "MContraction::A2AChromoMagneticOperatorField",
                  block=block, cacheBlock=cache_block, parities=parities,
                  left=left, right=right, gauge=gauge, output=output,
                  ifOrthogs=if_orthogs)


def load_nersc(name, file):
    return module(name, "MIO::LoadNersc", file=file)


def ape_smear(name, gauge, alpha, N, orthog_axis=3):
    return module(name, "MGauge::APESmear",
                  gauge=gauge, alpha=alpha, N=N, orthog_axis=orthog_axis)
