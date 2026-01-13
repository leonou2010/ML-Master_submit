import copy
import json
from pathlib import Path
from typing import Type, TypeVar, cast

import dataclasses_json
from search.journal import Journal

def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize ML-Master dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        node2best_local_node = {
            n.id: n.local_best_node.id  # type: ignore[attr-defined]
            for n in obj.nodes
            if getattr(n, "local_best_node", None) is not None
        }
        for n in obj.nodes:
            n.parent = None
            n.local_best_node = None
            n.child_count_lock = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["node2best_local_node"] = node2best_local_node
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to ML-Master dataclasses."""
    obj_dict = json.loads(s)

    if cls is Journal:
        # Journal stores MCTSNodes (subclass of Node). Deserialize accordingly so
        # resume mode can restore MCTS-specific fields (visits, stage, etc.).
        from search.mcts_node import MCTSNode  # local import to avoid cycles
        import threading

        nodes = [MCTSNode.from_dict(d) for d in obj_dict.get("nodes", [])]
        journal = Journal(nodes=nodes)

        id2nodes = {n.id: n for n in journal.nodes}
        for child_id, parent_id in obj_dict.get("node2parent", {}).items():
            if child_id in id2nodes and parent_id in id2nodes:
                id2nodes[child_id].parent = id2nodes[parent_id]
                id2nodes[child_id].__post_init__()

        for node_id, best_id in obj_dict.get("node2best_local_node", {}).items():
            if node_id in id2nodes and best_id in id2nodes:
                id2nodes[node_id].local_best_node = id2nodes[best_id]  # type: ignore[attr-defined]

        for n in journal.nodes:
            if getattr(n, "child_count_lock", None) is None:
                n.child_count_lock = threading.Lock()  # type: ignore[attr-defined]

        return cast(G, journal)

    obj = cls.from_dict(obj_dict)
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
