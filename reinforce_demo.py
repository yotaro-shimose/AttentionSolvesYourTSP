from gat.modules.models.encoder import Encoder
from gat.modules.models.decoder import PolicyDecoder
from gat.reinforce.reinforce import Reinforce
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='./models')
args = parser.parse_args()

load_dir = args.load
reinforce = Reinforce(load_dir=load_dir)
reinforce.demo()
