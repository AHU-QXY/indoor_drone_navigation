import os
import argparse
import numpy as np
import torch.optim
from copy import deepcopy
import config

from xuance import get_arguments
from xuance.common import space2shape
from envs import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.utils import ActivationFunctions

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe.")
    parser.add_argument("--method", type=str, default="ddpg")
    parser.add_argument("--env", type=str, default="drones")
    parser.add_argument("--env-id", type=str, default="DenseObstacles")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--running-steps", type=int, default=12000000)
    parser.add_argument("--parallels", type=int, default=8)
    parser.add_argument("--config", type=str, default="configs/ddpg/DenseObstacles.yaml")
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--test_episode", type=int, default=5)
    parser.add_argument("--model_folder", type=str, default='')

    return parser.parse_args()


def run(args):
    agent_name = args.agent
    set_seed(args.seed)

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)
    args.log_dir = os.path.join(args.log_dir, args.env_id)

    if args.test:
        # args.test_episode = 100
        args.parallels = 1
        args.render = True
        args.test_mode = True
        args.benchmark = 0
    # else:
    #     args.render = False

    # build environments
    envs = make_envs(args)
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space
    n_envs = envs.num_envs

    # prepare the Representation
    from xuance.torch.representations import Basic_Identical
    representation = Basic_Identical(input_shape=space2shape(args.observation_space),
                                     device=args.device)

    # prepare the Policy
    from xuance.torch.policies import DDPGPolicy
    policy = DDPGPolicy(action_space=args.action_space,
                        representation=representation,
                        actor_hidden_size=args.actor_hidden_size,
                        critic_hidden_size=args.critic_hidden_size,
                        initialize=None,
                        activation=ActivationFunctions[args.activation],
                        activation_action=ActivationFunctions[args.activation_action],
                        device=args.device)

    # prepare the Agent
    from xuance.torch.agents import DDPG_Agent, get_total_iters
    actor_optimizer = torch.optim.Adam(policy.actor_parameters, args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(policy.critic_parameters, args.critic_learning_rate)
    actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                           total_iters=get_total_iters(agent_name, args))
    critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.25,
                                                            total_iters=get_total_iters(agent_name, args))
    agent = DDPG_Agent(config=args,
                       envs=envs,
                       policy=policy,
                       optimizer=[actor_optimizer, critic_optimizer],
                       scheduler=[actor_lr_scheduler, critic_lr_scheduler],
                       device=args.device)  # 定义智能体

    # start running
    envs.reset()
    if args.benchmark:
        def env_fn():
            args_test = deepcopy(args)
            args_test.parallels = 1
            return make_envs(args_test)

        train_steps = args.running_steps // n_envs
        eval_interval = args.eval_interval // n_envs
        test_episode = args.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = agent.test(env_fn, test_episode)
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            agent.train(eval_interval)
            config.testmode = 0
            test_scores = agent.test(env_fn, test_episode)
            config.testmode = 1

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": agent.current_step}
                # save best model
                agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if not args.test:  # train the model
            n_train_steps = args.running_steps // n_envs
            agent.train(n_train_steps)
            agent.save_model("final_train_model.pth")
            print("Finish training!")
        else:  # test the model
            def env_fn():
                return envs

            agent.render = True
            agent.load_model(path=agent.model_dir_load, model=args.model_folder)  # 加载模型
            scores = agent.test(env_fn, args.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")

    # the end.
    envs.close()
    agent.finish()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)
