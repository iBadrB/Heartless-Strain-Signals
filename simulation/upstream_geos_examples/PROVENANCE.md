# Upstream GEOS example decks (KGD)

`kgdToughnessDominated_base.xml` and `kgdToughnessDominated_benchmark.xml` are the
unmodified example decks distributed with GEOS (`inputFiles/hydraulicFracturing` in the
GEOS repository, https://github.com/GEOS-DEV/GEOS/tree/develop/inputFiles/hydraulicFracturing)
from which every 2D KGD deck in this thesis descends.

Thesis modifications relative to these originals (contained in the packaged decks under
`inputs/kgd/`): mesh extents and gridding, perforation/source placement for the
spacing sweep, source-term scales for the injection-rate sweep ($Q_0/2^k$), run
durations (100 s / 150 s), and the time-history output collections
(`elementAperture`, `averageStrain`, and nodal `totalDisplacement`).

To see the exact changes for any deck:
    diff upstream_geos_examples/kgdToughnessDominated_base.xml <thesis deck>

The heterogeneous 3D family descends from the GEOS heterogeneous in-situ stress
benchmark; its upstream originals (decks, property tables, and table builder) are
archived in `heterogeneous/`.
In particular, the packaged `inputs/heterogeneous/tables/` folder comes from the upstream
GEOS `tables/` folder:
https://github.com/GEOS-DEV/GEOS/tree/develop/inputFiles/hydraulicFracturing/tables,
with the table builder at
https://github.com/GEOS-DEV/GEOS/blob/develop/inputFiles/hydraulicFracturing/tables/buildInputTables.py.
`buildInputTables.py` and every CSV are byte-identical to the originals. A verified diff against the thesis decks shows the mesh,
the property tables (byte-identical), and the pumping schedule are unchanged; the
thesis modifications are the added time-history output collections and cadences, the
parameterized fracture hydraulic-aperture model (used by the pre-existing-fracture
cases), and a Newton iteration cap raised from 40 to 80. The frac_20..23 decks add the
fiber displacement export and, for frac_22/23, the HFTS-2 `tables_hfts2` built by
`buildInputTables_hfts2.py` (compare with the upstream `buildInputTables.py`).

GEOS version pins are in ../GEOS_VERSIONS.md.
