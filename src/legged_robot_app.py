import random, sys, os, pygame
import numpy as np
from src.legged_robot import LeggedRobot
from pygame.locals import *
import gymnasium as gym
import time
import multiprocessing

class LeggedRobotApp(object):

    def __init__(self, genomes, config, env_name="Walker2d-v4", agent_name="LeggedRobot"):
        print(env_name)
        self.envs = []
        self.score = 0
        self.crash_info = []
        for index, genome in enumerate(genomes):
            env_id = f"{env_name}_{index}"  
            # create environment with different ids
            env=gym.make(env_name, render_mode="rgb_array")
            observation = env.reset()
            self.envs.append([env, observation])
            self.legged_robots = LeggedRobot(env.observation_space.shape[0],env.action_space.shape[0],genome,config)
        
    def reset_enviroments(self):
        for env in self.envs:
            env.reset()
    
    def step_enviroments(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []
        for env, ac in zip(self.envs, actions):
            ob, rew, terminated, truncated, info = env[0].step(ac)
            done = terminated or truncated
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                env[0].reset()

        return obs, rewards, dones, infos
    
    def play(self):
        while True:
            actions = []
            for index, env in enumerate(self.envs):
                
                action = self.legged_robots.compute_action(env[1])
                actions.append(action)
                obs, rew, truncated, terminated, info=env[0].step(action)
                done = terminated or truncated
                if done:
                    env[0].reset()
                    print("The environment", index, "has been reset")
                else:
                    env[1] = obs
            
            
            # self.obs, rewards, dones, infos = self.step_enviroments(actions)
            # if all(dones):
            #     break

    
                

    def on_loop(self):

        # =========----==========================================================
        """ CHECK FLAP """
        # =========----==========================================================
        for bird in self.birds:
            bird.flap_decision(self.pipes)
        # =========----==========================================================



        # =========----==========================================================
        """ CHECK CRASH """
        # =========----==========================================================
        for index, bird in enumerate(self.birds):
            if bird.check_crash(self.pipes, self.base.basex, self.score):
                self.crash_info.append((bird.crashInfo, bird.genome))
                del self.birds[index]
                if len(self.birds) == 0:
                    bird.specie_died = True
                    return True
        # =========----==========================================================


        # =========----==========================================================
        """ CHECK FOR SCORE """
        # =========----==========================================================
        break_one = break_two = False
        for bird in self.birds:
            playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
            for pipe in self.pipes.upper:
                pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    self.score += 1
                    break_one = break_two = True
                    SOUNDS['point'].play() if SOUND_ON else None
                if break_one:
                    break
            if break_two:
                break

        # =========----==========================================================


        # =========----==========================================================
        """ MOVE BASE """
        # =========----==========================================================
        self.base.move(self.birds)
        # =========----==========================================================


        # =========----==========================================================
        """ MOVE PLAYER """
        # =========----==========================================================
        for bird in self.birds:
            bird.move()
        # =========----==========================================================


        # =========----==========================================================
        """ MOVE PIPES """
        # =========----==========================================================
        self.pipes.move(self.birds)
        # =========----==========================================================
        return False


    def on_render(self):
        # =========----==========================================================
        """ DRAW BACKGROUND """
        # =========----==========================================================
        SCREEN.blit(IMAGES['background'], (0,0))
        # =========----==========================================================


        # =========----==========================================================
        """ DRAW PIPES """
        # =========----==========================================================
        self.pipes.draw(SCREEN)
        # =========----==========================================================


        # =========----==========================================================
        """ DRAW BASE """
        # =========----==========================================================
        SCREEN.blit(IMAGES['base'], (self.base.basex, BASEY))
        # =========----==========================================================


        # =========----==========================================================
        """ DRAW STATS """
        # =========----==========================================================
        disp_tools.displayStat(SCREEN, self.birds[0].distance*-1, text="distance")
        disp_tools.displayStat(SCREEN, self.score, text="scores")
        for bird in self.birds:
            SCREEN.blit(IMAGES['player'][bird.index], (bird.x, bird.y))
        # =========----==========================================================


        # =========----==========================================================
        """ UPDATE DISPLAY """
        # =========----==========================================================
        pygame.display.update()
        # =========----==========================================================


        # =========----==========================================================
        """ TICK CLOCK """
        # =========----==========================================================
        FPSCLOCK.tick(FPS)
        # =========----==========================================================


if __name__ == "__main__":
    flappy = FlappyBirdApp()
    flappy.play()
