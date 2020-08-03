import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_state


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, eps=0.0):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False, eps=eps)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        "eval_ep_cnt": len(eval_episode_rewards),
        "eval_reward": np.mean(eval_episode_rewards)
    }


def evaluate_same_env(actor_critic, ob_rms, num_processes, device, eval_envs=None, eps=0.0,
                      deterministic=False, eval_ep=10, max_steps=0, repeat_eps=1,
                      use_rand_actions=False):

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    step = 0
    if max_steps is None or max_steps <= 0:
        max_steps = np.inf

    if repeat_eps > 1:
        eps = eps / float(repeat_eps)
        rand_act = (torch.rand(num_processes, 1) < eps)
        rand_act_cnt = torch.zeros_like(rand_act).float()
    rand_actions = None

    while len(eval_episode_rewards) < eval_ep and step < max_steps:
        with torch.no_grad():
            if repeat_eps > 1:
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=deterministic, eps=eps, rand_action_mask=rand_act,
                    rand_actions=rand_actions
                )
                rand_act_cnt = (rand_act_cnt + 1) * rand_act
                new_rand_act = rand_act_cnt > repeat_eps

                cnt_new = new_rand_act.sum()
                if cnt_new > 0:
                    rand_act[new_rand_act] = False
                    new_rand_act[new_rand_act] = 0

                available = ~rand_act
                av_cnt = available.sum()
                if av_cnt > 0:
                    rand_act[available] = (torch.rand(av_cnt) < eps)

                if not use_rand_actions and rand_act.sum() > 0:
                    rand_actions = action[rand_act]
                else:
                    rand_actions = None
            else:
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=deterministic, eps=eps)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
        step += 1

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    if len(eval_episode_rewards) < eval_ep:
        return None

    return {
        # "eval_ep_cnt": len(eval_episode_rewards),
        "eval_reward": np.mean(eval_episode_rewards)
    }


def evaluate_first_ep(actor_critic, ob_rms, num_processes, device, eval_envs=None, eps=0.0):

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = dict({})

    # obs = eval_envs.reset()
    # Do not reset - take step
    obs, _, done, infos = eval_envs.step(
        torch.tensor([np.random.randint(eval_envs.action_space.n) for _ in range(num_processes)]).unsqueeze(1)
    )

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_processes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False, eps=eps)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for idx, info in enumerate(infos):
            if 'episode' in info.keys() and idx not in eval_episode_rewards:
                eval_episode_rewards[idx] = info['episode']['r']

    eval_envs.close()

    eval_episode_rewards = list(eval_episode_rewards.values())

    # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    #     len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        "eval_reward": eval_episode_rewards
    }
