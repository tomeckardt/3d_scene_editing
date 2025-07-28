from argparse import ArgumentParser
import json
from pathlib import Path
from random import sample

def generate_split(d: Path, train_ratio: float = 0.8, min_num_views: int = 15) -> tuple[dict, dict]:
    train_dict = {}
    test_dict = {}
    for scene in filter(lambda x: x.is_dir(), d.iterdir()):
        scene_dict_train = {}
        scene_dict_test = {}
        for room in scene.iterdir():
            for item in room.iterdir():
                views = sorted(int(f.stem) for f in item.iterdir() if f.suffix == ".hdf5")
                if len(views) < min_num_views:
                    continue
                views_train = sorted(sample(views, k=int(train_ratio * len(views))))
                views_test = [x for x in views if x not in views_train]
                item_name = room.name + "/" + item.name
                scene_dict_train[item_name] = views_train
                scene_dict_test[item_name] = views_test
        train_dict[scene.name] = scene_dict_train
        test_dict[scene.name] = scene_dict_test
    return train_dict, test_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="The path to your renderings")
    args = parser.parse_args()
    path = Path(args.path)
    dtrain, dtest = generate_split(path)
    with open(path / "selected_seqs_train.json", "w") as f:
        json.dump(dtrain, f)
    with open(path / "selected_seqs_test.json", "w") as f:
        json.dump(dtest, f)