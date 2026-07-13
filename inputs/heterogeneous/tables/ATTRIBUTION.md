# Attribution

The contents of this folder (`buildInputTables.py` and all CSV property/schedule
tables) are unmodified, byte-identical copies from the GEOS repository
(GEOS-DEV/GEOS, `inputFiles/hydraulicFracturing/tables`):

- Folder: https://github.com/GEOS-DEV/GEOS/tree/develop/inputFiles/hydraulicFracturing/tables
- Table builder: https://github.com/GEOS-DEV/GEOS/blob/develop/inputFiles/hydraulicFracturing/tables/buildInputTables.py

They are redistributed here so the packaged heterogeneous decks run as-is.
No file in this folder was edited for the thesis; the exact GEOS
commits used are pinned in `simulation/GEOS_VERSIONS.md`, and the full provenance of
the deck families is documented in `simulation/upstream_geos_examples/PROVENANCE.md`.

The HFTS-2-calibrated counterpart tables in `../tables_hfts2/` are thesis-built by
`buildInputTables_hfts2.py` from the B6S stress profile and are NOT upstream files.
