import pickle
import cloudpickle
import numpy as np
from multiprocessing import Process, Pipe

class AllDeadEnvsException(Exception):
	def __init__(self, message="All environments are dead."):
		self.message = message
		super().__init__(self.message)
		
def worker(remote, parent_remote, env_fn, id):
	start=0
	parent_remote.close()
	env,agent = env_fn()
	obs=env.reset()
	print("Processo id" ,id, "avviato")
	cmd = remote.recv()
	if(cmd=='Start'):
		while True:
				action = agent.compute_action(obs)
				obs, _, done, _ = env.step(action)
				if done:
					will = (agent.get_genome().key,env.get_total_fitness())
					remote.send(will)
					remote.close()
					break
				env.fitness()
	return

		
class CloudpickleWrapper(object):
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		self.x = pickle.loads(ob)
	
	def __call__(self):
		return self.x()


class SubprocVecEnv():
	def __init__(self, env_fns):
		self.waiting = False
		self.closed = False
		no_of_envs = len(env_fns)
		self.remotes, self.work_remotes = \
			zip(*[Pipe() for _ in range(no_of_envs)])
		self.ps = {}
		self.dict_remotes = {}
		self.dict_work_remotes = {}
		self.wills = {}

		for i in range(no_of_envs):
			wrk = self.work_remotes[i]
			rem = self.remotes[i]
			fn = env_fns[i]
			self.dict_remotes[i] = rem
			self.dict_work_remotes[i] = wrk 
			proc = Process(target = worker, 
				args = (wrk, rem, CloudpickleWrapper(fn),i))
			self.ps[i] = proc
		

		
		for p in self.ps.values():
			p.daemon = True
			p.start()
		try:
			for remote in self.dict_work_remotes.values():
				remote.close()
		except Exception as e:
			print(e)
		self.closed = True

	def remove_dead_envs(self, results):
		for result in results:
			done = result[2]
			index = result[4]
			if done:
				self.dict_remotes[index].send(('close', None))	
				will= self.dict_remotes[index].recv()
				self.ps[index].join()
				self.ps[index].close()
				self.ps.pop(index)
				self.dict_remotes.pop(index)
				self.dict_work_remotes.pop(index)
				self.wills[will[0]]=will[1]
			

	def get_no_of_envs(self):
		return len(self.dict_remotes.values())
	
	def start(self):
		for remote in self.dict_remotes.values():
			remote.send('start')
		
		results = [remote.recv() for remote in self.dict_remotes.values()]
		for index in len(self.ps):
			self.ps[index].join()
			self.ps[index].close()
			self.ps.pop(index)
			self.dict_remotes.pop(index)
			self.dict_work_remotes.pop(index)
		return results

