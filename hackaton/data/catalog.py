"""Loading and shaping of the product dataset."""

import ast
from typing import Dict, Iterator, Optional, Tuple

import pandas as pd

from hackaton import config


class ProductCatalog:
    """The product dataset, parsed once and shared by the UI and the search index.

    ``data/final_data.csv`` holds one column per category, each cell being a
    stringified Python dict describing a single advertisement. This class
    converts those cells back into dicts and exposes one flat DataFrame per
    category.
    """

    def __init__(self, csv_path=None):
        self.csv_path = csv_path or config.CSV_PATH

        # Load CSV into DataFrame
        raw = pd.read_csv(self.csv_path)

        # Convert any stringified dicts back to Python dicts
        for col in raw.columns:
            raw[col] = raw[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
            )
        self.raw = raw

        # Create a separate DataFrame per category
        self.frames: Dict[str, pd.DataFrame] = {
            category.key: pd.DataFrame(raw[category.csv_column].tolist())
            for category in config.CATEGORIES
        }

    def frame(self, key: str) -> pd.DataFrame:
        """Return the DataFrame for a category key ('car', 'laptop', 'phone')."""
        return self.frames.get(key, pd.DataFrame())

    def frame_for_ui_label(self, ui_label: str) -> pd.DataFrame:
        """Return the DataFrame matching a combo-box label, or an empty one."""
        for category in config.CATEGORIES:
            if category.ui_label == ui_label:
                return self.frames[category.key]
        return pd.DataFrame()

    def iter_documents(self) -> Iterator[Tuple[str, str, dict]]:
        """Yield ``(doc_id, doc_text, metadata)`` for every advertisement.

        Each row is flattened into a single searchable string; the metadata
        keeps the category and the original row index so a hit can be traced
        back to the CSV.
        """
        for category in config.CATEGORIES:
            for row_idx, row in self.frames[category.key].iterrows():
                doc_text = ' '.join(f"{k}: {v}" for k, v in row.items())
                metadata = {'category': category.key, 'row_idx': row_idx}
                yield f"{category.key}_{row_idx}", doc_text, metadata
