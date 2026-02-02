"""Visualization utilities for tutorials."""

from collections import defaultdict
from pathlib import Path

from openff.toolkit import Molecule
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

DISTINCT_COLORS = [
    (0.90, 0.10, 0.10),
    (0.10, 0.50, 0.90),
    (0.15, 0.75, 0.15),
    (0.95, 0.55, 0.00),
    (0.58, 0.15, 0.80),
    (0.00, 0.75, 0.75),
    (0.90, 0.20, 0.60),
    (0.55, 0.35, 0.00),
    (0.40, 0.75, 1.00),
    (0.80, 0.80, 0.00),
    (1.00, 0.40, 0.40),
    (0.00, 0.45, 0.35),
    (0.75, 0.50, 1.00),
    (0.00, 0.60, 0.30),
    (0.85, 0.65, 0.15),
    (0.35, 0.35, 0.80),
    (0.95, 0.75, 0.50),
]


def _generate_rdkit_imgs(
    molecules: list[Molecule],
    labels: list[list[str]],
    names: list[str],
    output_dir: str | Path,
    prefix: str = "",
    svg_size: tuple[int, int] = (800, 600),
    grid_size: tuple[int, int] = (500, 400),
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equiv: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, mol, lab in zip(names, molecules, labels, strict=False):
        for i, _atom in enumerate(mol.atoms):
            equiv[lab[i]].append((name, i))
    shared = {k for k, v in equiv.items() if len({t[0] for t in v}) > 1}

    color_map = {
        label: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        for i, label in enumerate(sorted(shared))
    }

    rdkit_mols, h_atoms_list, h_colors_list = [], [], []
    for mol, lab in zip(molecules, labels, strict=False):
        rdkit_mols.append(mol.to_rdkit())
        atoms = [i for i, lbl in enumerate(lab) if lbl in color_map]
        colors = {i: color_map[lab[i]] for i in atoms}
        h_atoms_list.append(atoms)
        h_colors_list.append(colors)

    saved = []

    grid_path = output_dir / f"{prefix}shared_labels.png"
    img = Draw.MolsToGridImage(
        rdkit_mols,
        molsPerRow=2,
        subImgSize=grid_size,
        legends=names,
        highlightAtomLists=h_atoms_list,
        highlightAtomColors=h_colors_list,
        highlightBondLists=[[] for _ in molecules],
    )
    img.save(str(grid_path))
    saved.append(grid_path)

    for name, rdmol, h_atoms, h_colors in zip(
        names, rdkit_mols, h_atoms_list, h_colors_list, strict=False
    ):
        drawer = rdMolDraw2D.MolDraw2DSVG(*svg_size)
        drawer.drawOptions().addAtomIndices = True
        drawer.DrawMolecule(
            rdmol,
            highlightAtoms=h_atoms,
            highlightAtomColors={i: (*c, 1.0) for i, c in h_colors.items()},
            highlightBonds=[],
            highlightAtomRadii={i: 0.4 for i in h_atoms},
        )
        drawer.FinishDrawing()
        svg_path = output_dir / f"{prefix}{name.lower()}_shared_labels.svg"
        svg_path.write_text(drawer.GetDrawingText())
        saved.append(svg_path)

    return saved
