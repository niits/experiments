from datetime import date
from typing import Annotated
from sqlalchemy import func
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from fastapi_mcp import FastApiMCP
from typing import List, Optional
from pydantic import BaseModel


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, description="Name of the hero")
    real_name: str | None = Field(default=None, description="Real name of the hero")
    origin: str | None = Field(default=None, description="Place of origin")
    superpowers: str | None = Field(default=None, description="Comma-separated list of superpowers")
    team: str | None = Field(default=None, description="Team or organization affiliation")
    first_appearance: str | None = Field(default=None, description="First appearance (comic, year, etc.)")
    biography: str | None = Field(default=None, description="Short biography or summary")
    dob: Optional[date] = Field(default=None, description="Date of birth")

class PaginatedHeroes(BaseModel):
    total: int
    offset: int
    limit: int
    count: int
    results: List[Hero]


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    try:
        exist_hero = session.exec(select(Hero).where(Hero.name == hero.name)).first()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error checking hero existence: {e}")

    if exist_hero:
        raise HTTPException(status_code=400, detail="Hero with this name already exists")

    try:
        session.add(hero)
        session.commit()
        session.refresh(hero)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating hero: {e}")
    return hero

@app.get(
    "/heroes/",
    response_model=PaginatedHeroes,
    description="Paginated list of heroes, max 100 per page, to avoid overloading the server"
)
def list_heroes(
    session: SessionDep,
    name: str | None = Query(None, min_length=3, description="Keyword to search in name"),
    secret_name: str | None = Query(None, min_length=3, description="Keyword to search in secret_name"),
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    query = select(Hero)
    if name:
        query = query.where(Hero.name.contains(name)) # type: ignore
    if secret_name:
        query = query.where(Hero.real_name.contains(secret_name)) # type: ignore
    
    total_query = select(func.count()).select_from(query.subquery())
    total = session.exec(total_query).one()
    
    heroes = session.exec(query.offset(offset).limit(limit)).all()
    return PaginatedHeroes(
        total=total,
        offset=offset,
        limit=limit,
        count=len(heroes),
        results=list(heroes),
    )


@app.get("/heroes/{hero_id}", description="Get hero by ID")
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero


@app.delete("/heroes/{hero_id}", description="Delete hero by ID")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}

mcp = FastApiMCP(app)

mcp.mount(mount_path="/sse")

mcp.setup_server()
