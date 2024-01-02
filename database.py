from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "mysql://root:@localhost/mitra-telkom"

metadata = MetaData()

user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("nik", String(16)),
    Column("email", String(250)),
    Column("password", String(250))
)

mitra = Table(
    "mitra",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("nof", String(250)),
    Column("st", String(250)),
    Column("po", String(250)),
    Column("dd", String(250)),
    Column("ddgr", String(250)),
    Column("la", String(250)),
    Column("jp", String(250)),
    Column("khs", String(250)),
    Column("alker", String(250)),
    Column("stok", String(250)),
    Column("tim", String(250)),
    Column("rapih", String(250)),
    Column("tanggal_upload", Date)
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={'connect_timeout': 120},
    pool_pre_ping=True
)

metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db

    finally:
        db.close()