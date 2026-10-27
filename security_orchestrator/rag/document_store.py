
import json
from pathlib import Path


class DocumentStore:
    """Load and validate cybersecurity reference documents from JSON."""

    REQUIRED_FIELDS = {"id", "title", "source", "content"}

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load_documents(self):
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Knowledge-base file not found: {self.file_path}"
            )

        with self.file_path.open("r", encoding="utf-8") as file:
            documents = json.load(file)

        if not isinstance(documents, list):
            raise ValueError(
                "Knowledge base must contain a JSON list."
            )

        seen_ids = set()

        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise ValueError(
                    f"Document at index {index} must be an object."
                )

            missing = self.REQUIRED_FIELDS - document.keys()

            if missing:
                raise ValueError(
                    f"Document at index {index} is missing fields: "
                    f"{sorted(missing)}"
                )

            for field in self.REQUIRED_FIELDS:
                if (
                    not isinstance(document[field], str)
                    or not document[field].strip()
                ):
                    raise ValueError(
                        f"Document at index {index} has an empty or "
                        f"invalid '{field}' field."
                    )

            if document["id"] in seen_ids:
                raise ValueError(
                    f"Duplicate document ID: {document['id']}"
                )

            seen_ids.add(document["id"])

        return documents
