from argparse import ArgumentParser
import json
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class FurnitureItem:
    aid: list
    jid: str
    uid: str
    sourceCategoryId: str | None = None
    valid: bool | None = None
    size: np.ndarray | None = None
    category: str | None = None
    title: str | None = None
    bbox: np.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.size, list):
            self.size = np.array(self.size)
        if isinstance(self.bbox, list):
            self.bbox = np.array(self.bbox)

    def serialize(self) -> dict:
        d = self.__dict__
        if d["size"] is not None:
            d["size"] = d["size"].tolist()
        if d["bbox"] is not None:
            d["bbox"] = d["bbox"].tolist()
        return d

def parse_furniture(scene: dict) -> list[FurnitureItem]:
    return [FurnitureItem(**item) for item in scene["furniture"]]

def furniture_to_json(furniture: list[FurnitureItem]) -> list[dict]:
    return [f.serialize() for f in furniture]

def replace_furniture(furniture: list[FurnitureItem], jid: str, replacement_jid: str):
    try:
        old_f = next(f for f in furniture if f.jid == jid)
    except:
        raise ValueError(f"No furniture with jid '{jid}' found.")
    old_f.jid = replacement_jid

def remove_furniture(furniture: list[FurnitureItem], jid: str) -> bool:
    try:
        old_f = next(f for f in furniture if f.jid == jid)
    except:
        return False
    furniture.remove(old_f)
    return True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_file", type=str, help="Name of the scene's JSON description")
    parser.add_argument(
        "--json_path", type=str, help="Path to the scene's JSON description",
        default="submodules/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT"
    )
    parser.add_argument("--replace", help="Replace furniture item with first jid with second jid", nargs="+")
    parser.add_argument("--remove", help="Remove furniture item(s) with jid", nargs="+")
    parser.add_argument("--out_dir", type=str, help="Output path for modified JSON", default="modified_scenes")
    args = parser.parse_args()
    with open(os.path.join(args.json_path, args.json_file)) as f:
        scene = json.load(f)
    furniture = parse_furniture(scene)
    if args.remove is not None:
        for jid in args.remove:
            remove_furniture(furniture, jid)
    if args.replace is not None:
        if len(args.replace) % 2 == 1:
            raise ValueError("Number of arguments for --replace must be even")
        for jid, rep_jid in [args.replace[i:i+2] for i in range(0, len(args.replace), 2)]:
            replace_furniture(furniture, jid, rep_jid)
    scene["furniture"] = furniture_to_json(furniture)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, args.json_file), "w") as f:
        json.dump(scene, f)

