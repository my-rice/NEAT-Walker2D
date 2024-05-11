from src.env import Environment, AvailableEnvironments
from src.legged_robot import LeggedRobot, AvailableAgents
import hydra
from omegaconf import MISSING, OmegaConf
import neat
# Create the environment
DEBUG = False



@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")
    assert AvailableEnvironments.has_value(cfg.env_name)
    assert AvailableAgents.has_value(cfg.agent_name)
    assert cfg.max_episodes > 0
    assert cfg.render is True or cfg.render is False
    env = Environment(cfg.env_name, cfg.neat_path, True, cfg.max_episodes) # to check if neat_path is correct

    observation_space_dim = env.get_observation_space()
    action_space_dim = env.get_action_space()
    # Create the legged robot agent

    env.run()
    # # Main loop
    # for i in range(cfg.max_episodes):
    #     observation = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.compute_action(observation)
    #         observation, reward, done, info = env.step(action)

    #         print("Fitness: ", env.fitness())
    #         #agent.update_agent(env.fitness())
    #         if(cfg.render):
    #             env.render()
    env.close()

if __name__ == "__main__":
    main()