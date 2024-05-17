import pickle
import cloudpickle
import numpy as np
from multiprocessing import Process, Pipe

class AllDeadEnvsException(Exception):
	def __init__(self, message="All environments are dead."):
		self.message = message
		super().__init__(self.message)
		
def worker(remote, parent_remote, env_fn, id):
	parent_remote.close()
	env,agent = env_fn()
	obs=env.reset()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			action = agent.compute_action(obs)
			obs, reward, done, info = env.step(action)
			env.fitness()
			remote.send((obs, reward, done, info, id))

		elif cmd == 'render':
			remote.send(env.render())

		elif cmd == 'close':
			will = (agent.get_genome().key,env.get_total_fitness())
			remote.send(will)
			remote.close()

			break

		elif cmd == 'reset':
			remote.send(env.reset())

		else:
			raise NotImplementedError
		
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
			

		# for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
		# 	print("Vicino")
		# 	print(wrk)
		# 	print(rem)
		# 	proc = Process(target = worker, 
		# 		args = (wrk, rem, CloudpickleWrapper(fn)))
		# 	self.ps.append(proc)

		# self.remotes, self.work_remotes = [list(t) for t in zip(*[Pipe() for _ in range(no_of_envs)])]
		# pipes = [Pipe() for _ in range(no_of_envs)]
		# self.remotes = {i: pipe for i, pipe in enumerate(pipes)}
		# self.work_remotes = {i: pipe for i, pipe in enumerate(pipes)}
		# self.ps = {}
		# for id_wrk, id_rem, fn in zip(self.work_remotes, self.remotes, env_fns):
		# 	proc = Process(target = worker, 
		# 		args = (self.work_remotes[id_wrk], self.remotes[id_rem], CloudpickleWrapper(fn),id_wrk))
		# 	self.ps[id_wrk] = proc

		
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
	
	def step_async(self):
		if self.waiting:
			raise Exception
		self.waiting = True
		for remote in self.dict_remotes.values():
			remote.send(('step', None))
	
	def step_wait(self):
		if not self.waiting:
			raise Exception
		self.waiting = False

		results = [remote.recv() for remote in self.dict_remotes.values()]
		obs, rews, dones, infos , ids= zip(*results)

		self.remove_dead_envs(results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos
	
	def step(self):
		if(len(self.dict_remotes)<=0):
			raise AllDeadEnvsException
		self.step_async()
		return self.step_wait()
	
	def reset(self):
		for remote in self.dict_remotes.values():
			remote.send(('reset', None))

		return np.stack([remote.recv() for remote in self.dict_remotes.values()])
	
	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.dict_remotes.values():
				remote.recv()
		for remote in self.dict_remotes.values():
			remote.send(('close', None))
		for p in self.ps.values():
			p.join()
		self.closed = True

	def return_wills(self):
		return self.wills