from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    paper_title: str
    year: int = 0
    venue: str = ""
    authors: str = ""
    institute_authors: str = ""
    institute_roles: str = ""
    departments: str = ""
    paper_url: str = ""
