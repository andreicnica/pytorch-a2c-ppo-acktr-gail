import time
from collections import deque

import numpy as np
import torch
import wandb
import csv
import os
from liftoff import OptionParser, dict_to_namespace
import yaml
from argparse import Namespace

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate_same_env

from models.model import Policy
from models import get_model

LOG_HEADER = {
    "update": None,
    "timesteps": None,
    "median": None,
    "reward": None,
    "min": None,
    "max": None,
    "dist_entropy": None,
    "value_loss": None,
    "action_loss": None,
}


def flatten_cfg(cfg: Namespace):
    lst = []
    for key, value in cfg.__dict__.items():
        if isinstance(value, Namespace):
            for key2, value2 in flatten_cfg(value):
                lst.append((f"{key}.{key2}", value2))
        else:
            lst.append((key, value))
    return lst


def parse_opts(check_out_dir: bool = True):
    """ This should be called by all scripts prepared by liftoff.

        python script.py results/something/cfg.yaml

        in your script.py

          if __name__ == "__main__":
              from liftoff import parse_opts()
              main(parse_opts())
    """

    opt_parser = OptionParser("liftoff", ["config_path", "session_id"])
    opts = opt_parser.parse_args()
    config_path = opts.config_path
    with open(opts.config_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    opts = dict_to_namespace(config_data)

    if not hasattr(opts, "out_dir"):
        opts.out_dir = f"results/experiment_{os.path.dirname(config_path)}"
        opts.run_id = 1
    if check_out_dir and not os.path.isdir(opts.out_dir):  # pylint: disable=no-member
        os.mkdir(opts.out_dir)
        print(f"New out_dir created: {opts.out_dir}")
    else:
        print(f"Existing out_dir: {opts.out_dir}")

    return opts


def run(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.model.recurrent:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    use_wandb = args.use_wandb
    eval_eps = args.eval_eps

    if use_wandb:
        experiment_name = f"{args.full_title}_{args.run_id}"
        from wandb_key import WANDB_API_KEY

        os.environ['WANDB_API_KEY'] = WANDB_API_KEY

        wandb.init(project="atari_ppo", name=experiment_name)
        wandb.config.update(dict(flatten_cfg(args)))

    if args.seed == 0:
        args.seed = args.run_id + 1

    print(f"SEED: {args.seed}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.out_dir

    os.environ['OPENAI_LOGDIR'] = args.out_dir
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'

    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    flog = open(log_dir + "/logs.csv", 'w')
    log_writer = csv.DictWriter(flog, LOG_HEADER.keys())
    log_writer.writeheader()

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, False)

    base_model = get_model(args.model, envs.observation_space.shape, envs.action_space)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_model,
        base_kwargs=args.model)
    actor_critic.to(device)

    print("Neural Network:")
    print(actor_critic)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    eval_episodes = args.eval_episodes
    eval_env_max_steps = 6000 * (eval_episodes // args.num_processes + 1)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {} | num timesteps {} | FPS {} | Last {} training episodes: mean/median "
                "reward {:.1f}/{:.1f} | min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            data_plot = {"update": j,
                         "timesteps": total_num_steps,
                         "reward": np.mean(episode_rewards),
                           "median": np.median(episode_rewards),
                           "min": np.min(episode_rewards),
                           "max":np.max(episode_rewards),
                           "dist_entropy": dist_entropy,
                           "value_loss": value_loss,
                           "action_loss": action_loss,
                           }

            log_writer.writerow(data_plot)

            if use_wandb:
                wandb.log(data_plot, step=total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)

            determinitistic = args.eval_determinitistic

            # Evaluate for
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes,
                                      args.num_processes,
                                      args.gamma, eval_log_dir, device, True)

            eval_info = evaluate_same_env(actor_critic, eval_envs, ob_rms, args.num_processes,
                                          device, deterministic=determinitistic, eps=eval_eps,
                                          eval_ep=eval_episodes, max_steps=eval_env_max_steps)
            eval_envs.close()

            # Evaluate for
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes,
                                      args.num_processes,
                                      args.gamma, eval_log_dir, device, True)

            eval_info_mode = evaluate_same_env(actor_critic, eval_envs, ob_rms, args.num_processes,
                                               device, deterministic=determinitistic, eps=0.,
                                               eval_ep=eval_episodes, max_steps=eval_env_max_steps)

            eval_envs.close()

            eval_inf = dict()

            if eval_info is not None and eval_info_mode is not None:
                for k, v in eval_info.items():
                    eval_inf[f"{k}_test"] = v

                for k, v in eval_info_mode.items():
                    eval_inf[f"{k}_training"] = v

                eval_inf["eval_gap"] = eval_info_mode["eval_reward"] - eval_info["eval_reward"]

            if use_wandb and len(eval_inf) > 0:
                wandb.log(eval_inf, step=total_num_steps)


if __name__ == "__main__":
    run(parse_opts())
