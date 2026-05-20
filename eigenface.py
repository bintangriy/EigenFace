import numpy as np
import cv2
from eigen import compute_eigen_manual

IMG_SIZE = 128

class EigenFaceRecognizer:
    def __init__(self, images, num_components=30):
        self.mean_face = np.mean(images, axis=1).reshape(-1, 1)

        self.A = images - self.mean_face

        self.C = np.dot(self.A.T, self.A)

        self.eigenvalues, self.eigenvectors_small = compute_eigen_manual(
            self.C,
            num_components
        )

        self.eigenfaces = np.dot(
            self.A,
            self.eigenvectors_small.T
        )

        self.eigenfaces = (
            self.eigenfaces /
            np.linalg.norm(self.eigenfaces, axis=0)
        )

        self.projected_images = np.dot(
            self.eigenfaces.T,
            self.A
        )

    def recognize(self, test_image):
        test_image = test_image.reshape(-1, 1)

        test_image = test_image - self.mean_face

        test_proj = np.dot(
            self.eigenfaces.T,
            test_image
        )

        distances = np.linalg.norm(
            self.projected_images - test_proj,
            axis=0
        )

        min_dist = np.min(distances)

        index = np.argmin(distances)

        return index, min_dist

    def get_eigenfaces(self, num_faces=5):
        faces = []

        for i in range(
            min(num_faces, self.eigenfaces.shape[1])
        ):
            face = self.eigenfaces[:, i].reshape(
                IMG_SIZE,
                IMG_SIZE
            )

            normalized = cv2.normalize(
                face,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            faces.append(normalized.astype(np.uint8))

        return faces