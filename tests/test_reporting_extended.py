import unittest
from unittest.mock import patch

from FeDa4Fair.utils.reporting import get_git_info, prep_info_dict


class TestReportingExtended(unittest.TestCase):
    @patch("FeDa4Fair.utils.reporting.subprocess.check_output")
    def test_get_git_info(self, mock_check_output):
        mock_check_output.side_effect = ["sha123\n", "url\n"]
        sha, url = get_git_info()
        self.assertEqual(sha, "sha123")
        self.assertEqual(url, "url")

    @patch("FeDa4Fair.utils.reporting.SOURCE_FILE")
    @patch("FeDa4Fair.utils.reporting.get_git_info")
    def test_prep_info_dict(self, mock_git, mock_source):
        mock_source.read_text.return_value = "[tag:tag1]body1[/tag]"
        mock_git.return_value = ("sha", "url")

        tags = prep_info_dict()
        self.assertIn("tag1", tags)
        self.assertEqual(tags["tag1"], ["body1"])
        self.assertEqual(tags["commit"], ["sha"])


if __name__ == "__main__":
    unittest.main()
