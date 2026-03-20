from pydantic import BaseModel


class ConceptDefinition(BaseModel):
    uri: str
    names_doc: str
    type: str | None = None
    names: list[str] = []
    cnt: int = 0
