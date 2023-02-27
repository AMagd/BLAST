import gym
# import gym_minigrid
import dreamerv2.api as dv2
import time
from common import natural_imgsource, envs
from matplotlib import pyplot as plt
import cv2
import os

def main():
    env_type = "original"
    random_bg = True # random images (True) - seuential video frames (False)

    config = dv2.defaults.update({
        'logdir': '~/logdir/minigrid/BLAST/1',
        'log_every': 1e3,
        'train_every': 10,
        'prefill': 1e5,
        'actor_ent': 3e-3,
        'loss_scales.kl': 1.0,
        'discount': 0.99,
        # 'add_recon_loss': False,
        # 'kl.balance': 1,
        'encoder.norm': 'batchnorm',
        'rssm.obs_out_norm': 'batchnorm',
        # 'ema': 0.99
    }).parse_flags()


    env = gym.make('MiniGrid-Dynamic-Obstacles-6x6-v0')

    if env_type == "background_mode":
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
    while True:
        obs = env.reset()
        for k in range(100):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            print(k)
            cv2.imshow('minigrid', cv2.resize(cv2.cvtColor(obs['image'], cv2.COLOR_BGR2RGB), (144, 144), interpolation = cv2.INTER_AREA))
            if cv2.waitKey(80) & 0xFF == ord('s'):
                break
    cv2.destroyAllWindows()


    # obs = env.reset()
    # obs['image']
    # dv2.train(env, config)

if __name__ == '__main__':
  main()