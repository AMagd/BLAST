import gym
# import gym_minigrid
import dreamerv2.api as dv2
import time
from common import natural_imgsource, envs
from matplotlib import pyplot as plt
import cv2
import os

def main():
    ###################### PARAMETERS TO CHANGE ######################
    model= 'SG' # [DreamerV2, recon, SG, EMA, BN, AR, BLAST]
    env_type = "smaller_agent" # [unmodified, video, random_frames, smaller_agent, color_direction]
    print(f'{" Model = " + model + " " :#^100}')
    print(f'{" ENV = " + env_type + " " :#^100}')
    ##################################################################

    # 1: normal minigrid env, 2: minigrid env with rewards 0 and 1, 3: minigrid env with rewards -1, 0 and 1
    config = dv2.defaults.update({
        'logdir': f'~/logdir/minigrid/{model}/{env_type}/3',
        'prefill': 1e4,
        'train_every': 1,
        'steps': 5e4,
        'pred_discount': False,
        'discount': 0.99,
        'rssm.hidden': 600,
        'rssm.deter': 600,
        'rssm.stoch': 16,
        'rssm.discrete': 64,
        'actor_grad': 'reinforce',
        'loss_scales.kl': 0.1,
        'loss_scales.reward': 100.0,
        'clip_rewards': 'tanh', # reward_clamp
        'kl.balance': 0.8,
        'log_every': 1e3,
        'actor_ent': 3e-3,
    }).parse_flags()

    if model == 'recon':
        config = config.update({
            'add_recon_loss': False
            }).parse_flags()
    
    elif model == 'SG':
        config = config.update({
            'add_recon_loss': False,
            'kl.balance': 1
            }).parse_flags()
    
    elif model == 'EMA':
        config = config.update({
            'add_recon_loss': False,
            'kl.balance': 1,
            'ema': 0.99
            }).parse_flags()
    
    elif model == 'BN':
        config = config.update({
            'add_recon_loss': False,
            'kl.balance': 1,
            'ema': 0.99,
            'encoder.norm': 'batchnorm',
            'rssm.obs_out_norm': 'batchnorm'
        }).parse_flags()
    
    elif model == 'AR':
        config = config.update({
            'add_recon_loss': False,
            'kl.balance': 1,
            'ema': 0.99,
            'encoder.norm': 'batchnorm',
            'rssm.obs_out_norm': 'batchnorm',
            'rssm.ar_steps': 3
        }).parse_flags()
    
    else:
        pass


    env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0')
    env = envs.MiniGridMDPWrapper(env)
    random_bg = env_type=='random_frames' # random images (True) - seuential video frames (False)

    if env_type == "video" or env_type == "random_frames":
        mode = 'train'
        bg_path = config.bg_path_train if mode == 'train' else config.bg_path_test
        files = [os.path.join(bg_path, f) for f in os.listdir(bg_path) if os.path.isfile(os.path.join(bg_path, f))]
        _bg_source = natural_imgsource.RandomVideoSource(shape=[48, 48], filelist=files, random_bg=random_bg, max_videos=100, grayscale=False)
        env = envs.BackgroundWrapper(env, _bg_source)
    elif env_type == "color_direction":
        env = envs.ColorDirectionWrapper(env)
    elif env_type == "smaller_agent":
        env = envs.SmallerAgentWrapper(env)
    else:
        env = envs.RGBImgObsWrapper(env)

    
    # obs = env.reset()
    # while True:
    #     obs = env.reset()
    #     for k in range(100):
    #         action = env.action_space.sample()
    #         obs, *_ = env.step(action)
    #         print(k)
    #         cv2.imshow('minigrid', cv2.resize(cv2.cvtColor(obs['image'], cv2.COLOR_BGR2RGB), (144, 144), interpolation = cv2.INTER_AREA))
    #         if cv2.waitKey(80) & 0xFF == ord('s'):
    #             break
    # cv2.destroyAllWindows()


    obs = env.reset()
    dv2.train(env, config)

if __name__ == '__main__':
  main()