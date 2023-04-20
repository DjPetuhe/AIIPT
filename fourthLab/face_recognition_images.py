import cv2
import utils
import pickle

with open("encodings.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)

imageJolie = cv2.imread("examples/Angelina_Jolie_0020.jpg")
imageEmma = cv2.imread("examples/Emma_Watson_1.jpg")
imageMe = cv2.imread("examples/Me_1.jpg")
images = [imageJolie, imageEmma, imageMe]

for i in range(len(images)):
    encodings = utils.face_encodings(images[i])
    names = []
    for encoding in encodings:
        counts = {}
        for (name, encodings) in name_encodings_dict.items():
            counts[name] = utils.nb_of_matches(encodings, encoding)
        if all(count == 0 for count in counts.values()):
            name = "Unknown"
        else: name = max(counts, key=counts.get)
        names.append(name)
    for rect, name in zip(utils.face_rects(images[i]), names):
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(images[i], name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("image", images[i])
    cv2.waitKey(0)

