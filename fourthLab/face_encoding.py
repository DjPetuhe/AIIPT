import os
import cv2
import utils
import pickle

root_dir = 'dataset'
class_names = os.listdir(root_dir)
image_paths = utils.get_image_paths(root_dir, class_names)
name_encoding_dict = {}

nb_current_image = 1
for image_path in image_paths:
    print(f"Image processed {nb_current_image}/{len(image_paths)}")
    image = cv2.imread(image_path)
    encodings = utils.face_encodings(image)
    name = image_path.split(os.path.sep)[-2]
    e = name_encoding_dict.get(name, [])
    e.extend(encodings)
    name_encoding_dict[name] = e
    nb_current_image += 1

with open("encodings.pickle", "wb") as f:
    pickle.dump(name_encoding_dict, f)
