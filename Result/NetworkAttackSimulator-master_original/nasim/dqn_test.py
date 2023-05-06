"""Script for running dqn

Usage
$ python -m nasim.dqn_test scenario
For example:
$ python -m nasim.dqn_test tiny.yaml
"""

import os.path as osp

import nasim
from nasim.agents.dqn_agent import DQNAgent

DQN_DIR = osp.dirname(osp.abspath(__file__))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "NASim Customization for dqn"
        )
    )
    parser.add_argument("env_name", type=str,
                        help="scenario name")
    parser.add_argument("policy_name", type=str,
                        help="policy name")
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[128],
                        help="(default=[64. 64])")
    args = parser.parse_args()
    print(args.env_name)    
    print(args.policy_name)
    
    SCENARIO_FILE = osp.join(DQN_DIR,args.env_name)
    POLICY_FILE=osp.join(DQN_DIR, args.policy_name)
    # SCENARIO_FILE = osp.join(DQN_DIR,'tiny.yaml')
    # POLICY_FILE = osp.join(DQN_DIR, 'dqn_tiny.dqn_tiny.pt')
    env=nasim.generate(5, 3, num_os=3)
    env = nasim.load(SCENARIO_FILE,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True)
    env.render_network_graph(show=True)

    line_break = f"\n{'-'*60}"
    print(line_break)
    print("Using AI policy")
    print(line_break)
    dqn_agent = DQNAgent(env, verbose=False, **vars(args))
    dqn_agent.load(POLICY_FILE)
    ret, steps, goal = dqn_agent.run_eval_episode(
        env, True, 0.01, "readable"
    )
    # ret, steps, goal = dqn_agent.run_eval_episode(
    #     env
    # )
    env.render_network_graph(show=True)
    print(line_break)
    print(f"Episode Complete")
    print(line_break)
    if goal:
        print("Goal accomplished. Sensitive data retrieved!")
    print(f"Final Score={ret}")
    print(f"Steps taken={steps}")
