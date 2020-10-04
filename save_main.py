from multiprocessing import Process, Queue, Pipe
from agents.actor import Actor
from agents.learner import Learner
if __name__ == "__main__":
    actor = Actor()
    process_list = []
    for _ in range(5):
        process_list.append(Process(actor.start()))

    learner = Learner()
