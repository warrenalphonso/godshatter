import pathlib

import pytest
from antibody_mcts.pdb import Antibody, Mutation, get_sequences, make_mutation, score_antibody


@pytest.fixture
def faspr_executable():
    return pathlib.Path("/Users/wba/repos/FASPR/FASPR")

def test_get_sequences(glucagon_pdb_file, cr3022_pdb_file):
    "Test correct parsing of a glucagon PDB file."
    assert get_sequences(glucagon_pdb_file) == {"A": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"}

    assert get_sequences(cr3022_pdb_file) == {
        "C": "TNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP",
        "H": "QMQLVQSGTEVKKPGESLKISCKGSGYGFITYWIGWVRQMPGKGLEWMGIIYPGDSETRYSPSFQGQVTISADKSINTAYLQWSSLKASDTAIYYCAGGSGISTPMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC",
        "L": "DIQLTQSPDSLAVSLGERATINCKSSQSVLYSSINKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECS",
    }

def test_score_antibody(cr3022):
    assert isinstance(score_antibody(cr3022), float)

"""
- test that after mutate, I can read PDB file and get expected chains
- test that diff of PDB files is small and in expected region
"""

def test_make_mutation(cr3022, faspr_executable):
    "Test that make_mutation makes the correct mutation and has expected changes to PDB file."
    # A really good mutation that BeAtMuSiC found: H position 97 G → C gave delta delta G of -1.7.
    mutation = Mutation(chain="H", position=97, new_aa="C")

    result = make_mutation(faspr_executable, antibody=cr3022, mutation=mutation)
    assert result.H != cr3022.H
    assert result.H == "QMQLVQSGTEVKKPGESLKISCKGSGYGFITYWIGWVRQMPGKGLEWMGIIYPGDSETRYSPSFQGQVTISADKSINTAYLQWSSLKASDTAIYYCACGSGISTPMDVWGQGTTVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
    assert result.L == cr3022.L == "DIQLTQSPDSLAVSLGERATINCKSSQSVLYSSINKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECS"

    """$ diff tests/data/relaxed-6w41.pdb tests/data/c2dde259711bc0c128f8c1c05545436c.pdb
759,762c759,764
< ATOM      1  N   GLY H  94     -55.709 -28.995  14.683  1.00  0.00           N
< ATOM      2  CA  GLY H  94     -54.678 -29.079  13.671  1.00  0.00           C
< ATOM      3  C   GLY H  94     -53.521 -29.987  14.030  1.00  0.00           C
< ATOM      4  O   GLY H  94     -52.879 -29.816  15.072  1.00  0.00           O
---
> ATOM      1  N   CYS H  94     -55.709 -28.995  14.683  1.00  0.00           N
> ATOM      2  CA  CYS H  94     -54.678 -29.079  13.671  1.00  0.00           C
> ATOM      3  C   CYS H  94     -53.521 -29.987  14.030  1.00  0.00           C
> ATOM      4  O   CYS H  94     -52.879 -29.816  15.072  1.00  0.00           O
> ATOM      5  CB  CYS H  94     -54.118 -27.689  13.360  1.00  0.00           C
> ATOM      6  SG  CYS H  94     -52.840 -27.681  12.083  1.00  0.00           S
"""
