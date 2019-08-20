import unittest
import sys
sys.path.append("../")
import feedforward_robust as ffr

def make_model():
    model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma = sigma)
    return model

class TestRobustMLP(unittest.TestCase):

    def test_regular_training(self):
        model = make_model()
        sess = tf.Session()
        model.fit(sess, X, y, lr, epochs = 1, reg = 0)

    def test_norms(self):
        pass

    def test_distance(self):
        pass

    def test_fgsm_attack(self):
        pass

    def test_pgd_attack(self):
        pass


if __name__ == "__main__":
    unittest.main()
