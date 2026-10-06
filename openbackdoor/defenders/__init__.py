from .defender import Defender
from .cube_defender import CUBEDefender, CasualCUBEDefender
from .graceful_defender import GraCeFulDefender
from .svd_defender import SVDDefender
from .onion_defender import ONIONDefender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .new3_defender import New3Defender

DEFENDERS = {
    "base": Defender,
    'cube': CUBEDefender,
    'casualcube': CasualCUBEDefender,
    'graceful': GraCeFulDefender,
    'svd': SVDDefender,
    'onion': ONIONDefender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'new3': New3Defender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
