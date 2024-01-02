from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from sqlalchemy.sql import func
import json
from datetime import datetime
from json import JSONEncoder


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    nik = Column(String(16))
    email = Column(String(250))
    password = Column(String(250))

class Mitra(Base):
    __tablename__ = "mitra"

    id = Column(Integer, primary_key=True, index=True)
    nof = Column(String(250), nullable=True)
    st = Column(String(250), nullable=True)
    po = Column(String(250), nullable=True)
    dd = Column(String(250), nullable=True)
    ddgr = Column(String(250), nullable=True)
    la = Column(String(250), nullable=True)
    jp = Column(String(250), nullable=True)
    khs = Column(String(250), nullable=True)
    alker = Column(String(250), nullable=True)
    stok = Column(String(250), nullable=True)
    tim = Column(String(250), nullable=True)
    rapih = Column(String(250), nullable=True) 
    tanggal_upload = Column(DateTime, default=func.now())
    
