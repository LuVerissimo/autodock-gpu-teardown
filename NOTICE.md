# Notice

This repository documents an independent performance study of AutoDock-GPU by Lu Verissimo. It is not affiliated
with, reviewed by, or endorsed by Scripps Research, the Forli lab, or the AutoDock-GPU developers. AutoDock is a
trademark of the Scripps Research Institute.

- `scripts/`: GPL-2.0-or-later (see the GNU GPL v2 text in the AutoDock-GPU repository).
- `patches/`: diffs against AutoDock-GPU files; each change carries the license of the file it modifies
  (GPL-2.0-or-later or LGPL-2.1-or-later). `ceiling-noatomic-WRONG-OUTPUT.diff` produces incorrect docking results
  by design and exists only to time an upper bound.
- Report text and data (README.md, data/, the release tarball of .dlg files): CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- The 42-complex test set (`diogomart/AD-GPU_set_of_42`, GPL-3.0) is not redistributed here.
