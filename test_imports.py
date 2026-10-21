import unittest
import face_swap

class FaceSwapTests(unittest.TestCase):
    def test_face_swap_function_exists(self):
        self.assertTrue(callable(face_swap.face_swap))

    def test_missing_image_raises_clear_error(self):
        with self.assertRaises(ValueError):
            face_swap.load_image("definitely_missing_file_12345.jpg")

if __name__ == "__main__":
    unittest.main()
