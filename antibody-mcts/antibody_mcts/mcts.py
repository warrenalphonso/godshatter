import dataclasses
import itertools
import json
import logging
import math
import pathlib
import random

from Bio.Data import IUPACData
from pybktree import BKTree
from prodigy_prot.modules.parsers import parse_structure
from prodigy_prot.predict_IC import calculate_ic

from antibody_mcts.pdb import Antibody, Mutation, make_mutation, score_antibody

logging.getLogger().setLevel(10000)

def _subs(a: Antibody, b: Antibody):
    return sum(ca != cb for ca, cb in zip(itertools.chain(a.H, a.L), itertools.chain(b.H + b.L)))

def get_chains(a: Antibody):
    return f"{a.H}-{a.L}"

def get_ic_network(pdb: pathlib.Path):
    structure_object, num_chains, num_residues = parse_structure(pdb.resolve())
    selection = {"H": 0, "L": 0, "C": 1}
    return calculate_ic(struct=structure_object, d_cutoff=5.5, selection=selection)

def get_unique_aa_ic_network(pdb: pathlib.Path):
    network = get_ic_network(pdb)
    return list(set(
        (r.parent.id, r.id[1])
        for r in itertools.chain(*network)
        if r.parent.id != "C"
    ))

@dataclasses.dataclass
class Node:
    antibody: Antibody
    parent: "Node | None"

@dataclasses.dataclass(frozen=True)
class SerializedAntibody:
    filename: str
    total_score: float
    visits: int

class MCTS:
    def __init__(self, *, iterations, depth, exploration_constant = math.sqrt(2), faspr_executable, mutations_dir):
        self.bktree = BKTree(distance_func=_subs)
        self.fname_to_state: dict[str, Antibody] = {}

        self.synced_states: dict[str, SerializedAntibody] = {}

        self.iterations = iterations
        self.search_depth = depth
        self.exploration_constant = exploration_constant
        self.random_mutation_attempts = 10
        self.faspr_executable = faspr_executable
        assert self.faspr_executable.exists()
        self.mutations_dir = mutations_dir
        assert self.mutations_dir.exists()

    def run(self, *, root_node: Node):
        self.bktree.add(root_node.antibody)
        self.fname_to_state[root_node.antibody.pdb.name] = root_node.antibody

        node = root_node
        for _ in range(self.iterations):
            leaf_node = self.select(node)
            reward = self.simulate(leaf_node)
            self.backprop(leaf_node, reward)

    def select(self, node: Node) -> Node:
        "Try creating a new leaf or pick one using UCT."
        for _ in range(self.search_depth):
            if unseen_antibody := self._unseen_random_mutation(node.antibody):
                self.bktree.add(unseen_antibody)
                self.fname_to_state[unseen_antibody.pdb.name] = unseen_antibody
                return Node(antibody=unseen_antibody, parent=node)
            node = self.uct(node=node)
        return node

    def simulate(self, node: Node) -> float:
        """
        Search at some depth, returning binding affinity of final random node as score for leaf node.

        Does this make sense? We're scoring a mutation by scoring the affinity of it + 5 more random mutations.
        """
        curr = node.antibody
        for _ in range(self.search_depth):
            curr = self._random_mutation(curr)
        return score_antibody(curr)

    def backprop(self, node: Node, reward: float): 
        while node:
            node.antibody.total_score += reward
            node.antibody.visits += 1
            node = node.parent

    def uct(self, node: Node) -> Node:
        "Select a child node using UCT."
        children = self._children(node=node)
        log_parent_visits = math.log(node.antibody.visits) if node.antibody.visits > 0 else 0
        def uct_score(antibody):
            exploration_term = self.exploration_constant * math.sqrt(log_parent_visits / antibody.visits)
            average_score = antibody.total_score / antibody.visits
            return -average_score + exploration_term
        best_antibody = max(children, key=uct_score)
        return Node(antibody=best_antibody, parent=node)

    def dumps_diff(self) -> list[dict]:
        "Serialize and dump diff between last load and current state."
        diff = []
        for antibody in self.fname_to_state.values():
            fname = antibody.pdb.name
            current = SerializedAntibody(filename=fname, total_score=antibody.total_score, visits=antibody.visits)

            if fname not in self.synced_states:
                diff.append(current)
            elif (synced_state := self.synced_states[fname]) != current:
                diff.append(SerializedAntibody(filename=fname, total_score=current.total_score - synced_state.total_score, visits=current.visits - synced_state.visits))
            self.synced_states[fname] = current

        return [dataclasses.asdict(o) for o in diff]

    def loads_diff(self, diff: list[dict]):
        "Load state diff from `diff`."
        serialized_diff = [SerializedAntibody(**d) for d in diff]
        for serialized_antibody in serialized_diff:
            if serialized_antibody.filename not in self.fname_to_state:
                antibody = Antibody(pdb=self.mutations_dir / serialized_antibody.filename, total_score=serialized_antibody.total_score, visits=serialized_antibody.visits)
                self.fname_to_state[serialized_antibody.filename] = antibody
                self.bktree.add(antibody)
            else:
                self.fname_to_state[serialized_antibody.filename].total_score += serialized_antibody.total_score
                self.fname_to_state[serialized_antibody.filename].visits += serialized_antibody.visits

            self.synced_states[serialized_antibody.filename] = serialized_antibody

    def _unseen_random_mutation(self, start: Antibody) -> Antibody | None:
        "Try random mutations until we get a new Antibody or hit the attempt limit."
        # Check if unseen mutation exists
        seen_neighbors = [get_chains(s[1]) for s in self.bktree.find(start, n=1)]
        residues_to_mutate = get_unique_aa_ic_network(start.pdb)
        n_possible_neighbors = len(residues_to_mutate) * 20 # 20 possible amino acids
        if len(seen_neighbors) < n_possible_neighbors:
            for _ in range(self.random_mutation_attempts):
                end = self._random_mutation(start, residues_to_mutate)
                if get_chains(end) not in seen_neighbors:
                    return end

    def _random_mutation(self, start: Antibody, residues=None) -> Antibody:
        "Using cached `residues` to avoid re-reading the file... Saves ~0.2 seconds."
        if residues is None:
            residues = get_unique_aa_ic_network(start.pdb)
        chain, position = random.choice(residues)
        new_aa = random.choice(list(IUPACData.protein_letters_1to3.keys()))
        mutation = Mutation(chain=chain, position=position, new_aa=new_aa)
        return make_mutation(self.faspr_executable, start, mutation)

    def _children(self, start: Antibody) -> list[Antibody]:
        "Explored antibodies with at most 1 substitution from `start`."
        return [self.fname_to_state[s[1].pdb.name] for s in self.bktree.find(start, n=1)]
