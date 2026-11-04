"""Resolve local README links for the PyPI package description."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict[str, Any]) -> None:
        readme = (Path(self.root) / "README.md").read_text(encoding="utf-8")
        image_root = "https://raw.githubusercontent.com/balamuruganpg/verascan/main/screenshots/"
        readme = readme.replace('src="screenshots/', f'src="{image_root}')
        readme = readme.replace('href="screenshots/', f'href="{image_root}')
        readme = readme.replace("](screenshots/", f"]({image_root}")
        readme = readme.replace('href="#', 'href="https://github.com/balamuruganpg/verascan#')
        metadata["readme"] = {"content-type": "text/markdown", "text": readme}
