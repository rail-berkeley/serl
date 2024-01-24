from .continuous.bc import BCAgent
from .continuous.sac import SACAgent

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
}
