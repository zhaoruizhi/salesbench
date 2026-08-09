from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from salesbench.goldbank.cohort import select_goldbank_cohort  # noqa: E402


class GoldBankCohortTest(unittest.TestCase):
    def records(self):
        records = []
        idx = 0
        for product in ["食品", "家居家电", "服饰", "个护美妆", "其他"]:
            for fan in ["头部", "头腰", "腰部", "尾部"]:
                for offset in range(3):
                    idx += 1
                    records.append(
                        {
                            "video_id": f"v{idx}",
                            "product_bucket": product,
                            "fan_segment": fan,
                            "douyin_handle": f"creator_{idx}",
                            "has_video_asset": 1,
                            "title": "title",
                            "video_text": "asr",
                        }
                    )
        return records

    def test_anchor_ids_are_preserved_first(self):
        config = select_goldbank_cohort(self.records(), total=10, anchor_ids=["v8", "v1"], seed=1)

        self.assertEqual(config["video_ids"][:2], ["v8", "v1"])

    def test_balances_product_fan_cells_before_repeating(self):
        config = select_goldbank_cohort(self.records(), total=20, anchor_ids=[], seed=1)
        lookup = {record["video_id"]: record for record in self.records()}
        cells = {(lookup[video_id]["product_bucket"], lookup[video_id]["fan_segment"]) for video_id in config["video_ids"]}

        self.assertEqual(len(cells), 20)

    def test_prefers_unique_creators(self):
        records = self.records()
        records[1]["douyin_handle"] = records[0]["douyin_handle"]
        config = select_goldbank_cohort(records, total=20, anchor_ids=[], seed=1)
        lookup = {record["video_id"]: record for record in records}
        creators = [lookup[video_id]["douyin_handle"] for video_id in config["video_ids"]]

        self.assertEqual(len(creators), len(set(creators)))

    def test_output_is_runner_config(self):
        config = select_goldbank_cohort(self.records(), total=5, anchor_ids=[], seed=1)

        self.assertEqual(config["version"], "evidence-alpha-v2")
        self.assertEqual(config["prompt_version"], "evidence-prompt-v7")
        self.assertEqual(config["schema_version"], "evidence-dataset-schema-v2")
        self.assertEqual(config["frames_per_video"], 16)
        self.assertEqual(len(config["video_ids"]), 5)


if __name__ == "__main__":
    unittest.main()
