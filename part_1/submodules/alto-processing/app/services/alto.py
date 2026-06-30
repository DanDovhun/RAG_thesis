"""Integration helpers for AltoProcessor."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.main_processor import AltoProcessor  # type: ignore


@dataclass
class ProcessingRequest:
    uuid: str
    width: int = 800
    height: int = 1200


class AltoProcessingError(RuntimeError):
    pass


class AltoProcessingService:
    def __init__(self) -> None:
        self._processor = AltoProcessor()

    def process(self, params: ProcessingRequest) -> str:
        alto_xml = self._processor.get_alto_data(params.uuid)
        if not alto_xml:
            raise AltoProcessingError("Nepodařilo se stáhnout ALTO data.")
        formatted = self._processor.get_formatted_text(
            alto_xml,
            params.uuid,
            params.width,
            params.height,
        )
        if not formatted:
            raise AltoProcessingError("Procesor vrátil prázdný výsledek.")
        return formatted
