"""Utilities to interact with PDB files for antibodies."""
import dataclasses
import hashlib
import pathlib
import subprocess
import tempfile
import typing

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from prodigy_prot.modules.parsers import parse_structure
from prodigy_prot.predict_IC import Prodigy

_parser = PDBParser(QUIET=True)

def get_sequences(pdb: pathlib.Path) -> dict[chr, str]:
    "Get amino acid sequence for each chain."
    structure = _parser.get_structure(pdb.name, pdb)
    sequences = {}
    for chain in structure.get_chains():
        chain_seq = []
        for residue in chain.get_residues():
            if not is_aa(residue.resname, standard=True):
                raise ValueError(f"Non-amino acid residue found: {residue.resname} ({seq1(residue.resname)})")

            aa = seq1(residue.resname)
            chain_seq.append(aa)

        sequences[chain.id] = "".join(chain_seq)
    return sequences

@dataclasses.dataclass
class Antibody:
    """
    This class should store all relevant state information for a version of the antibody.

    The MCTS Node will have some state of its own, but that should only be relevant to the
    particular search (eg parent for backprop).
    """
    # Heavy chain amino acid sequence
    H: str
    # Light chain
    L: str

    pdb: pathlib.Path

    total_score: float
    visits: int

    def __init__(self, *, pdb: pathlib.Path, total_score: float = 0.0, visits: int = 0):
        self.pdb = pdb
        sequences = get_sequences(pdb)
        self.H = sequences["H"]
        self.L = sequences["L"]
        self.total_score = total_score
        self.visits = visits

def score_antibody(antibody: Antibody) -> float:
    "Estimate binding affinity of antibody to antigen using PRODIGY."
    structure_object, num_chains, _num_residues = parse_structure(antibody.pdb)
    assert num_chains == 3
    predictor = Prodigy(structure_object, selection=["H,L", "C"])
    predictor.predict()
    return predictor.kd_val

def get_id(*, h: str, l: str):
    return hashlib.md5(f"{h}:{l}".encode()).hexdigest()

@dataclasses.dataclass(frozen=True)
class Mutation:
    chain: typing.Literal["H", "L"]
    position: int
    new_aa: chr

def make_mutation(faspr_executable: pathlib.Path, antibody: Antibody, mutation: Mutation) -> Antibody:
    """
    Create a new Antibody by applying `mutation` to `antibody`. We use FASPR to mutate.

    This creates the new PDB file in the same directory as `antibody.pdb`.
    The filename is set to `get_id(h, l)`.
    """
    dir = antibody.pdb.parent

    sequences = get_sequences(antibody.pdb)
    sequences[mutation.chain] = sequences[mutation.chain][:mutation.position] + mutation.new_aa + sequences[mutation.chain][mutation.position+1:]

    with tempfile.TemporaryDirectory() as temp_dir:
        sequences_path = pathlib.Path(temp_dir) / "sequence.txt"
        sequences_path.write_text("".join(sequences.values()))

        output_file = dir / f"{get_id(h=sequences['H'], l=sequences['L'])}.pdb"

        # Call FASPR
        command = [
            faspr_executable.resolve(),
            "-i",
            antibody.pdb.resolve(),
            "-o",
            output_file.resolve(),
            "-s",
            sequences_path.resolve()
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise ValueError(f"Something went wrong. {result.stderr=} {result.stdout=}")

    return Antibody(pdb=output_file)
