import face_recognition


def recognize_face(face_encodings, known_data):
    matches = face_recognition.compare_faces(known_data["encodings"], face_encodings)
    name = "Unknown"

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = known_data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    return name


def get_face_encodings(rgb_image, box):
    top, right, bottom, left = int(box[1]), int(box[2]), int(box[3]), int(box[0])
    encodings = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])
    return encodings
