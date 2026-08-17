"""
Per-job vector pool: lazily emits the Load/Smear modules needed to produce
a given A2A vector array, caching the resulting module name so two roles
that happen to want the same array within one job share one set of
modules instead of loading it twice.

combined(flavor, vw, hits) loads low modes (once, if the flavor has any)
followed by each hit in `hits`' high modes, concatenated into one array
via MIO::LoadCombinedA2AVecs -- base(flavor, hit, vw) is just the
single-hit case of this. Strange/charm have no low modes
(config.FLAVOR_HAS_LOW), so their arrays are high-modes-only.

loop_leg(flavor, hit, vw) is for an EMF quark loop's V or W leg: it looks
for a combined() array already cached in this job that covers `hit` for
that (flavor, vw) and, if found, builds a MUtilities::VectorPackRefSlice
pointer view into the (low-mode segment + that hit's high-mode segment)
of it -- no re-load. If nothing cached covers it, it falls back to a
fresh single-hit combined() load, pointer-packed the same way, so the
caller (and LoopPropagatorPtr on the Grid side) sees a uniform pointer
array either way. In practice light/strange V always hits the cache
(h_loop is always within the batched external hit range); charm's V and
every flavor's W never do, since nothing else populates a cached array
for them.
"""
import config
import modules as M


class VectorPool:
    def __init__(self, job):
        self.job = job
        self._combined = {}  # (flavor, vw, hits-tuple) -> module name
        self._smeared = {}   # (flavor, hit, vw, width_tag) -> module name
        self._loop_ptr = {}  # (flavor, hit, vw) -> pointer-pack module name

    def combined(self, flavor, vw, hits):
        """Full A2A vector array for (flavor, vw) covering exactly `hits`:
        low modes (if any) once, then each hit's high modes in order."""
        hits = tuple(hits)
        key = (flavor, vw, hits)
        if key in self._combined:
            return self._combined[key]

        has_low = config.FLAVOR_HAS_LOW[flavor]
        low_filestem = config.low_filestem(flavor, vw) if has_low else ""
        high_extensions = [f"{flavor}{h}_{vw}" for h in hits]
        name = f"a2a_{flavor}_{vw}_" + "".join(f"h{h}" for h in hits)
        self.job.add(M.load_combined_a2a_vecs(
            name, low_filestem, config.N_LOW if has_low else 0,
            f"{config.VW_BASE}/", high_extensions, config.N_HIGH,
            config.LOW_BIN_SIZE, config.HIGH_BIN_SIZE))
        self._combined[key] = name
        return name

    def base(self, flavor, hit, vw):
        """Full A2A vector array for (flavor, hit, vw) -- single-hit case
        of combined()."""
        return self.combined(flavor, vw, [hit])

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

    def _find_containing(self, flavor, vw, hit):
        """A cached combined() array covering `hit` for (flavor, vw), if
        any -- (module name, its hits tuple)."""
        for (f, v, hits), name in self._combined.items():
            if f == flavor and v == vw and hit in hits:
                return name, hits
        return None

    def loop_leg(self, flavor, hit, vw):
        """Pointer view of (low modes + `hit`'s high modes) for one leg of
        an EMF loop propagator, referencing an already-cached combined()
        array if one covers `hit`, otherwise a fresh single-hit load."""
        key = (flavor, hit, vw)
        if key in self._loop_ptr:
            return self._loop_ptr[key]

        found = self._find_containing(flavor, vw, hit)
        if found is not None:
            source, hits = found
        else:
            source = self.combined(flavor, vw, [hit])
            hits = (hit,)

        has_low = config.FLAVOR_HAS_LOW[flavor]
        idx = hits.index(hit)
        offsets, counts = [], []
        if has_low:
            offsets.append(0)
            counts.append(config.N_LOW)
        offsets.append((config.N_LOW if has_low else 0) + idx * config.N_HIGH)
        counts.append(config.N_HIGH)

        name = f"ptr_{flavor}{hit}_{vw}_loop"
        self.job.add(M.vector_pack_ref_slice(name, source, offsets, counts))
        self._loop_ptr[key] = name
        return name
