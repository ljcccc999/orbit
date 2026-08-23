import unittest

from orbit.config import OrbitConfig
from orbit.worlds import generate_world, split_seeds


class ConfigTests(unittest.TestCase):
    def test_presets_are_valid_and_scale(self):
        tiny = OrbitConfig.preset("tiny")
        seven = OrbitConfig.preset("7b")
        self.assertGreater(seven.rough_parameter_count(), 6_000_000_000)
        self.assertLess(seven.rough_parameter_count(), 9_000_000_000)
        self.assertGreater(seven.rough_parameter_count(), tiny.rough_parameter_count())

    def test_invalid_head_geometry_fails(self):
        with self.assertRaises(ValueError):
            OrbitConfig(width=255, heads=8, state_width=255)


class WorldTests(unittest.TestCase):
    def test_action_changes_only_named_object(self):
        example = generate_world(7)
        before = dict(example.state_before)
        after = dict(example.state_after)
        moving = example.action.split()[1]
        for color in before:
            if color != moving:
                self.assertEqual(before[color], after[color])

    def test_split_has_no_overlap(self):
        train, valid = split_seeds(100)
        self.assertTrue(set(train).isdisjoint(valid))


if __name__ == "__main__":
    unittest.main()
