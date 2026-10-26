from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import fetch_yfinance


class LoadAssetFilesTests(unittest.TestCase):
    def test_loads_only_the_supplemental_asset_files_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            data_dir.mkdir()
            for index, (file_name, _market_segment) in enumerate(
                fetch_yfinance.SUPPLEMENTAL_ASSET_FILES
            ):
                with (data_dir / file_name).open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["ticker", "company"])
                    writer.writerow([f"{index}01A.T", file_name])

            with (
                patch.object(fetch_yfinance, "DATA_DIR", data_dir),
                patch.object(fetch_yfinance, "PROJECT_ROOT", root),
            ):
                members, manifest = fetch_yfinance.load_asset_files(
                    fetch_yfinance.SUPPLEMENTAL_ASSET_FILES
                )

        self.assertEqual(len(members), 3)
        self.assertEqual(
            {member["market_segment"] for member in members},
            {"growth", "prime", "standard"},
        )
        self.assertEqual(
            {item["file_name"] for item in manifest},
            {file_name for file_name, _ in fetch_yfinance.SUPPLEMENTAL_ASSET_FILES},
        )


if __name__ == "__main__":
    unittest.main()
