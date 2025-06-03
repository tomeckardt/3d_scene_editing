import cv2
import h5py
import json
import numpy as np
import os

def render_views_with_uid_for_file(file: h5py.File, modified_file: h5py.File, uid: str, view_name: str):
    inst_map = json.loads(file["instance_attribute_maps"][()].decode("utf-8")) # type:ignore
    try:
        found_idx = next(i["idx"] for i in inst_map if i["uid"] == uid)
        if found_idx not in np.array(file["instance_segmaps"]):
            return
    except:
        return
    colors = np.array(file["colors"])
    colors_modified = np.array(modified_file["colors"])
    comparison = np.concatenate((colors, colors_modified))
    cv2.imwrite(os.path.join("comparison", view_name.replace(".hdf5", ".jpg")), comparison)

def render_views_with_uid(dir: str, modified_dir: str, uid: str):
    for render in filter(lambda x: x.endswith(".hdf5"), os.listdir(dir)): # TODO: Do this in parallel if needed
        file_path = os.path.join(dir, render)
        modified_file_path = os.path.join(modified_dir, render)
        if not os.path.exists(modified_file_path):
            continue
        file = h5py.File(file_path)
        modified_file = h5py.File(modified_file_path)
        render_views_with_uid_for_file(file, modified_file, uid, render)

if __name__ == "__main__":
    dir = "renderings/0a9c667d-033d-448c-b17c-dc55e6d3c386/"
    modified_dir = "renderings/0a9c667d-033d-448c-b17c-dc55e6d3c386_modified/"
    render_views_with_uid(dir, modified_dir, "9748/model")
    