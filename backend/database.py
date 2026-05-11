from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

DATABASE_URL = "sqlite:///./visionguard.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

class IntruderLog(Base):

    __tablename__ = "intruder_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    person_name = Column(String)

    detection_time = Column(String)

    screenshot = Column(String)

    alert_status = Column(String)

Base.metadata.create_all(bind=engine)