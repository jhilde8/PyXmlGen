"""
Per-job vector pool: lazily emits the Load/Smear modules needed to produce
a given A2A vector array, caching the resulting module name so two roles
that happen to want the same array within one job share one set of
modules instead of loading it twice.

combined(flavor, vw, hits) produces one array covering exactly `hits`:
low modes (once, if the flavor has any -- config.FLAVOR_HAS_LOW, or as
forced either way by with_low) followed by the high-mode block per hit.
base(flavor, hit, vw) is just the single-hit case. The two sides now take
different paths:

  V: MIO::LoadCombinedA2AVecsV reads each hit's expanded high-mode files
     from disk, with the 1/nHit hit-average normalization applied to the
     high blocks (nHit = len(hits), so single-hit jobs are numerically
     unchanged). The whole hit-average factor lives on the V side by
     convention.

  W: dense/combined representation -- the raw noise ComplexFields are
     loaded (MIO::LoadTimeDilutedSpinColorDiagonalNoise, one filestem per
     hit, order = hit order) and expanded in-job by MIO::LoadCombinedA2AVecsW
     into config.N_SC = 12 dense fermion fields per hit instead of
     config.N_HIGH expanded ones: slot = nLow + hit*12 + sc. W stays raw
     unit-modulus noise, no normalization. Downstream consumers must pair
     dense W slot (hit, sc) at timeslice t with expanded mode
     (hit*nt + t)*12 + sc.

The old loop_leg() (per-hit pointer views for EMF quark loops) is gone
with the EMF generators: the EMF rewrite does the hit average of the loop
propagator inside the job from full multi-hit arrays, so per-hit slicing
has no remaining consumer.
"""
import config
import modules as M


class VectorPool:
    def __init__(self, job):
        self.job = job
        self._combined = {}  # (flavor, vw, hits-tuple) -> module name
        self._smeared = {}   # (flavor, hit, vw, width_tag) -> module name
        self._noise = {}     # (flavor, hits-tuple) -> noise module name

    @staticmethod
    def _tag(hits, tag):
        # Names spell out the hit list unless the caller supplies a shorter
        # tag (e.g. "h8" for all eight hits). Only the name changes; the cache
        # keys stay on the hit list.
        return tag if tag else "".join(f"h{h}" for h in hits)

    def noise(self, flavor, hits, tag=None):
        """Raw-noise object for (flavor, hits), one file stem per hit; hit
        order of every array expanded from it is this `hits` order."""
        hits = tuple(hits)
        key = (flavor, hits)
        if key in self._noise:
            return self._noise[key]
        name = f"noise_{flavor}_{self._tag(hits, tag)}"
        stems = [config.noise_filestem(flavor, h) for h in hits]
        self.job.add(M.load_time_diluted_noise(name, stems,
                                               config.N_NOISE_PER_STEM))
        self._noise[key] = name
        return name

    def combined(self, flavor, vw, hits, with_low=None, tag=None, exact=False):
        """Full A2A vector array for (flavor, vw) covering exactly `hits`:
        low modes (if any) once, then each hit's high-mode block. V is read
        expanded from disk (with 1/len(hits) on the high blocks); W is the
        dense noise expansion, 12 fields per hit.

        with_low overrides config.FLAVOR_HAS_LOW for this array: pass False
        to build a high-mode-only array for a flavor that does have low
        modes. That is what a genuine tiling of the mode index needs -- the
        low block belongs to exactly one tile, so every other tile asks for
        its flavor's high modes alone. Suppressing it renames the array
        '..._nolow' so it can coexist with the full one in the same job.
        Passing True for a flavor with no low modes is an error rather than
        a silent no-op. Note this does not touch the 1/nHit factor, which is
        still len(hits) -- a tile of a larger calculation needs the global
        hit count there, not the tile's.

        exact reads the high modes from the exact solve tree instead of the
        sloppy one, for the AMA correction hit. It applies to V only -- W is
        the raw noise, identical in both sets, so one W array serves both
        accuracies -- and renames the array '..._exact'. The low-mode block is
        the same eigen-derived files either way."""
        hits = tuple(hits)
        if exact and vw != "v":
            raise ValueError(
                f"exact=True for a '{vw}' array: only V has an exact solve, "
                f"W is the shared noise")
        default_low = config.FLAVOR_HAS_LOW[flavor]
        has_low = default_low if with_low is None else bool(with_low)
        if has_low and not default_low:
            raise ValueError(
                f"with_low=True for flavor '{flavor}', which has no low modes "
                f"(config.FLAVOR_HAS_LOW)")
        key = (flavor, vw, hits, has_low, exact)
        if key in self._combined:
            return self._combined[key]

        low_filestem = config.low_filestem(flavor, vw) if has_low else ""
        suffix = "" if has_low == default_low else "_nolow"
        suffix += "_exact" if exact else ""
        name = f"a2a_{flavor}_{vw}_{self._tag(hits, tag)}{suffix}"
        if vw == "w":
            self.job.add(M.load_combined_a2a_vecs_w(
                name, config.LOW_BIN_SIZE, low_filestem,
                config.N_LOW if has_low else 0, self.noise(flavor, hits, tag)))
        else:
            high_extensions = [config.high_extension(flavor, h, vw, exact)
                               for h in hits]
            self.job.add(M.load_combined_a2a_vecs_v(
                name, low_filestem, config.N_LOW if has_low else 0,
                config.high_stem(exact), high_extensions, config.N_HIGH,
                config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE, n_hit=len(hits)))
        self._combined[key] = name
        return name

    def base(self, flavor, hit, vw, with_low=None, exact=False):
        """Full A2A vector array for (flavor, hit, vw) -- single-hit case
        of combined()."""
        return self.combined(flavor, vw, [hit], with_low=with_low, exact=exact)

    def smeared(self, flavor, hit, vw, width_tag, alpha, N):
        """Smeared version of base(flavor, hit, vw) at the given width.
        Caller must have already added a 'gauge_APE' module to the job."""
        key = (flavor, hit, vw, width_tag)
        if key in self._smeared:
            return self._smeared[key]
        base_name = self.base(flavor, hit, vw)
        name = f"a2a_{flavor}{hit}_{vw}_{width_tag}"
        self.job.add(M.a2a_covariant_smear(
            name, a2a_vectors=base_name, gauge="gauge_APE", alpha=alpha, N=N,
            orthog_axis=config.ORTHOG_AXIS, output="", multi_file=False))
        self._smeared[key] = name
        return name

