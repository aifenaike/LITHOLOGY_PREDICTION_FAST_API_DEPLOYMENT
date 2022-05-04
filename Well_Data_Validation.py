from pydantic import BaseModel
import pickle
# Class which describes Log measurements from well.
class Well_data(BaseModel):
    CALI: float
    DEPTH_MD: float
    DRHO: float
    DTC: float
    GR: float
    NPHI: float
    PEF: float
    RDEP: float
    RHOB: float
    RMED: float
    SP: float
    GROUP: str
    WELL: str
    FORMATION: str






