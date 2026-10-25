"""
One factory function per Hadrons module type used by the job generators.
Each function's keyword arguments match that module's Serializable Par
struct field-for-field (verified against the .hpp sources under
Hadrons/Hadrons/Modules/), so a typo'd or missing argument is a Python
TypeError at generation time instead of silently-wrong XML.
"""
from hadrons_xml import module


def load_combined_a2a_vecs_v(name, low_filestem, n_low, high_stem, high_extensions,
                            high_size, n_hit=0, io_split=""):
    # n_hit: 1/nHit hit-average normalization on the high-mode blocks; the
    # factor lives on the V side by convention, so set it (to the number of
    # hits in the job's estimator) only for V loads. 0 or 1 = no rescaling
    # (W loads, single-hit data).
    # io_split: split-grid read layout, space separated ("2 2 2 1"); empty =
    # every file read on the full grid.
    return module(name, "MIO::LoadCombinedA2AVecsV",
                  lowFilestem=low_filestem, nLow=n_low, highStem=high_stem,
                  highExtensions=high_extensions, highSize=high_size, nHit=n_hit,
                  ioSplit=io_split)


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


def load_combined_a2a_vecs_w(name, low_filestem, n_low, noise, io_split=""):
    # Dense/combined high-mode W representation: nLow low modes from files,
    # then 12 dense fermion fields per hit expanded from `noise`
    # (slot = nLow + hit*12 + sc, sc fastest). No nHit normalization: W stays
    # raw noise. io_split as for the V loader; only the low modes are read.
    return module(name, "MIO::LoadCombinedA2AVecsW",
                  lowFilestem=low_filestem, nLow=n_low, noise=noise,
                  ioSplit=io_split)


def a2a_high_mode_v(name, noise, action, solver, output, init_vec=0, n_vec=0):
    # init_vec/n_vec: compute noise vectors [init_vec, init_vec + n_vec) of
    # the hit, to split a hit across jobs or resume one; n_vec=0 means through
    # the last vector. Defaults compute the whole hit.
    return module(name, "MSolver::A2AHighModeV",
                  noise=noise, action=action, solver=solver, output=output,
                  initVec=init_vec, nVec=n_vec)


def a2a_low_modes_coarse(name, n_basis, eigen_pack, action, output):
    return module(name, f"MUtilities::A2ALowModesCoarse{n_basis}",
                  eigenPack=eigen_pack, action=action, output=output)


def a2a_low_modes(name, action, eigen_pack, output):
    return module(name, "MUtilities::A2ALowModes",
                  action=action, eigenPack=eigen_pack, output=output)


def vector_pack_ref_slice(name, source, offsets, counts):
    return module(name, "MUtilities::FermionVectorPackRefSlice",
                  source=source, offsets=offsets, counts=counts)


def a2a_covariant_smear(name, a2a_vectors, gauge, alpha, N, orthog_axis=3,
                         output="", multi_file=False):
    return module(name, "MUtilities::A2ACovariantSmear",
                  a2aVectors=a2a_vectors, gauge=gauge, alpha=alpha, N=N,
                  orthog_axis=orthog_axis, output=output, multiFile=multi_file)


def a2a_coarse_grid(name, fine, block_size, offsets, output, mode_offset=0,
                    mode_size=0, scale=1.0, io_split=""):
    # mode_offset/mode_size: the block of `fine` to sparsen (size 0 = to the
    # end). scale multiplies the coarse fields, e.g. n_hit to undo the V
    # loader's 1/nHit; always emitted, since a missing node would read as 0.
    return module(name, "MUtilities::A2ACoarseGrid",
                  fine=fine, modeOffset=mode_offset, modeSize=mode_size,
                  scale=scale, blockSize=block_size, offsets=offsets,
                  output=output, ioSplit=io_split)


def a2a_meson_field(name, left_block, right_block, left, right, output, gammas,
                    mom, time_slice_io=False):
    # left_block/right_block: GEMM operand sizes per side (see config.py).
    # time_slice_io: skip the temporal all-gather and write one file per
    # (momentum, gamma, global timeslice), named "<ioname>.tNNNN.h5", spread
    # over min(nt, nRank) writers instead of nmom*ngamma of them.
    return module(name, "MContraction::A2AMesonField",
                  leftBlock=left_block, rightBlock=right_block,
                  left=left, right=right, output=output, gammas=gammas,
                  mom=mom, timeSliceIO=time_slice_io)

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


def a2a_loop_new(name, left, right, n_low, block, input_loop=""):
    # Superset of a2a_loop: takes either W representation (deduced from the
    # array sizes and logged), blocks the mode sum so device residency is set
    # by `block` rather than by the mode count, and for dense W compresses
    # each hit's expanded V block to N_SC fields before contracting.
    # n_low must match the low-mode block both loaders were given.
    #
    # input_loop names a previous A2ALoopNew's output to seed the sum with
    # instead of zero. Chaining these is how a job that cannot hold every mode
    # at once accumulates one loop across separately loaded mode blocks; the
    # VM keeps only the link in flight and its predecessor alive. Each block
    # carries whatever normalization its loader applied, so pass nHit on the
    # high-mode V loads and leave the low-mode loads alone.
    return module(name, "MContraction::A2ALoopNew",
                  left=left, right=right, nLow=n_low, block=block,
                  inputLoop=input_loop)


def write_prop(name, prop, file, format="IEEE64BIG"):
    # Raw lexicographic BinaryIO write (no header, no checksum verification);
    # the module appends ".<traj>" to `file`. `format` must match whatever
    # MIO::LoadProp / an offline reader is told to expect.
    return module(name, "MIO::WriteProp", prop=prop, file=file, format=format)


def load_prop(name, file, format="IEEE64BIG"):
    return module(name, "MIO::LoadProp", file=file, format=format)


def a2a_extended_meson_field(name, left_block, right_block, types, left, right,
                              output, gammas1, gammas2, loop="",
                              loop_vw1="", loop_vw2="", time_slice_io=False):
    # The quark loop comes in exactly one of two ways -- `loop`, naming a
    # precomputed PropagatorField (the production path), or `loop_vw1`/
    # `loop_vw2`, naming the loop's vector arrays for in-module construction.
    # The latter is expanded-W only and exists as the verification reference;
    # it goes away with the corresponding branch in the module.
    #
    # time_slice_io also skips SumRing's temporal gather, so each rank holds
    # only its own t slab -- not just a different file layout.
    if bool(loop) == bool(loop_vw1 or loop_vw2):
        raise ValueError(f"module '{name}': pass either loop= or "
                         f"loop_vw1=/loop_vw2=, not both or neither")
    if not loop and not (loop_vw1 and loop_vw2):
        raise ValueError(f"module '{name}': the vector path needs both "
                         f"loop_vw1 and loop_vw2")
    return module(name, "MContraction::A2AExtendedMesonField",
                  leftBlock=left_block, rightBlock=right_block, types=types,
                  left=left, right=right, loop=loop,
                  loop_vw1=loop_vw1, loop_vw2=loop_vw2,
                  output=output, gammas1=gammas1, gammas2=gammas2,
                  timeSliceIO=time_slice_io)


def a2a_chromomagnetic_operator_field(name, left_block, right_block, parities,
                                       left, right, gauge, output, if_orthogs,
                                       time_slice_io=False):
    # See a2a_extended_meson_field on time_slice_io.
    return module(name, "MContraction::A2AChromoMagneticOperatorField",
                  leftBlock=left_block, rightBlock=right_block, parities=parities,
                  left=left, right=right, gauge=gauge, output=output,
                  ifOrthogs=if_orthogs, timeSliceIO=time_slice_io)


def load_nersc(name, file):
    return module(name, "MIO::LoadNersc", file=file)


# --- CG: actions, deflation, solvers, high-mode V ---------------------------

def gauge_single_precision_cast(name, field):
    return module(name, "MUtilities::GaugeSinglePrecisionCast", field=field)


def mobius_dwf(name, gauge, Ls, mass, M5, b, c, boundary, twist, single=False):
    # single=True is the MobiusDWFF registration, for the inner solve of a
    # mixed-precision CG; hand it the single-precision gauge field.
    return module(name, "MAction::MobiusDWFF" if single else "MAction::MobiusDWF",
                  gauge=gauge, Ls=Ls, mass=mass, M5=M5, b=b, c=c,
                  boundary=boundary, twist=twist)


def scaled_dwf(name, gauge, Ls, mass, M5, scale, boundary, twist, single=False):
    return module(name, "MAction::ScaledDWFF" if single else "MAction::ScaledDWF",
                  gauge=gauge, Ls=Ls, mass=mass, M5=M5, scale=scale,
                  boundary=boundary, twist=twist)


def load_coarse_eigenpack_200f(name, filestem, multi_file, size_fine,
                               size_coarse, red_black, Ls, block_size,
                               orthogonalise, gauge_xform=""):
    # block_size is a std::string in the Par struct ("4 4 4 4 12"), not a
    # vector: a list here would serialize to <elem> children and fail to parse.
    return module(name, "MIO::LoadCoarseFermionEigenPack200F",
                  filestem=filestem, multiFile=multi_file, sizeFine=size_fine,
                  sizeCoarse=size_coarse, redBlack=red_black, Ls=Ls,
                  blockSize=block_size, orthogonalise=orthogonalise,
                  gaugeXform=gauge_xform)


def coarse_deflation_blas_200f(name, eigen_pack, size):
    return module(name, "MGuesser::CoarseDeflationBLAS200F",
                  eigenPack=eigen_pack, size=size)


def deflation_src_cast(name, guesser):
    # Double-precision guesser wrapping a single-precision one: casts the
    # source down, guesses, casts the guess back up.
    return module(name, "MGuesser::DeflationSrcCastEPackF", guesser=guesser)


def mixed_precision_rb_prec_cg(name, inner_action, outer_action, max_inner,
                               max_outer, residual, if_cgd, inner_guesser="",
                               outer_guesser=""):
    return module(name, "MSolver::MixedPrecisionRBPrecCG",
                  innerAction=inner_action, outerAction=outer_action,
                  maxInnerIteration=max_inner, maxOuterIteration=max_outer,
                  residual=residual, innerGuesser=inner_guesser,
                  outerGuesser=outer_guesser, ifCGD=if_cgd)


def mixed_precision_rb_prec_cg_batched(name, inner_action, outer_action,
                                       max_inner, max_outer, max_patchup,
                                       residual, update_residual, if_cgd,
                                       inner_guesser="", outer_guesser=""):
    # The split-grid layout comes from --batched-solver-split on the command
    # line, not from here.
    return module(name, "MSolver::MixedPrecisionRBPrecCGBatched",
                  innerAction=inner_action, outerAction=outer_action,
                  maxInnerIteration=max_inner, maxOuterIteration=max_outer,
                  maxPatchupIteration=max_patchup, residual=residual,
                  innerGuesser=inner_guesser, outerGuesser=outer_guesser,
                  updateResidual=update_residual, ifCGD=if_cgd)


def a2a_high_mode_batched_v(name, noise, action, solver, output, init_vec=0,
                            n_vec=0, batch_size=0):
    # batch_size sources per solver call (0 or 1 = unbatched); must divide the
    # number of vectors this run computes. Each V is written at
    # <output>.<traj>/elem<i>.bin with i the noise vector index, so ranges of
    # one hit computed in different runs land in the same directory.
    return module(name, "MSolver::A2AHighModeBatchedV",
                  noise=noise, action=action, solver=solver, output=output,
                  initVec=init_vec, nVec=n_vec, batchSize=batch_size)


def ape_smear(name, gauge, alpha, N, orthog_axis=3):
    return module(name, "MGauge::APESmear",
                  gauge=gauge, alpha=alpha, N=N, orthog_axis=orthog_axis)
