from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import argparse
import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

BOUNDING_BOX_COLOR = "blue"
BOUNDING_BOX_WIDTH = 4
TEXT_COLOR = "white"
TEXT_FONT = ImageFont.load_default(size=48)

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings =  []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> Image:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"

        # print(name, bounding_box)
        _display_face(draw, bounding_box, name)

    del draw
    return pillow_image


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom),
        name,
        font=TEXT_FONT
    )
    # Adjust text position
    text_left = text_left + 16
    text_top = text_top - 12
    text_right = text_right + 16
    text_bottom = text_bottom + 4
    # Add padding
    rectange_left = text_left - 16
    rectangle_top = text_top
    rectange_right = text_right + 16
    rectange_bottom = text_bottom
    draw.rectangle(
        ((rectange_left, rectangle_top), (rectange_right, rectange_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
        width=BOUNDING_BOX_WIDTH
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
        font=TEXT_FONT
    )

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--analyze", action="store_true", help="Analyze an image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",

)
parser.add_argument(
    "-f", action="store", help="Path to an image to analyze"
)

args = parser.parse_args()


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.analyze:
        recognize_faces(image_location=args.f, model=args.m)
