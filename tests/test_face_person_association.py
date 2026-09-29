import numpy as np

from face_association import face_associated_with_person


def test_valid_face_associated_to_person():
    person_box = [40, 40, 220, 380]
    face_box = [90, 80, 160, 145]

    assert face_associated_with_person(face_box, [person_box], (480, 640)) is not None


def test_false_face_rejected_when_no_person_matches():
    person_box = [40, 40, 220, 380]
    face_box = [500, 80, 570, 145]

    assert face_associated_with_person(face_box, [person_box], (480, 640)) is None


def test_face_in_lower_body_region_is_rejected():
    person_box = [40, 40, 220, 380]
    face_box = [90, 250, 160, 320]

    assert face_associated_with_person(face_box, [person_box], (480, 640)) is None


def test_multiple_people_associate_to_the_correct_person_track():
    person_a = [30, 40, 200, 350]
    person_b = [300, 40, 470, 350]
    face_for_a = [100, 90, 160, 150]
    face_for_b = [360, 90, 420, 150]

    assert np.allclose(face_associated_with_person(face_for_a, [person_a, person_b], (480, 640)), person_a)
    assert np.allclose(face_associated_with_person(face_for_b, [person_a, person_b], (480, 640)), person_b)


def test_face_touching_image_edge_is_rejected():
    person_box = [50, 50, 220, 380]
    face_box = [2, 90, 58, 150]

    assert face_associated_with_person(face_box, [person_box], (480, 640)) is None
