import os.path as osp
import nasim
from nasim.agents.ql_agent import TabularQFunction , TabularQLearningAgent
QL_DIR = osp.dirname(osp.abspath(__file__))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=10000,
                        help="training steps (default=10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("-e", "--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    SCENARIO_FILE = osp.join(QL_DIR,args.env_name)
    env = nasim.load(SCENARIO_FILE)
    ql_agent = TabularQLearningAgent(
        env, verbose=args.quite,  **vars(args)
    )
    ql_agent.train()
    line_break = f"\n{'-'*60}"
    print(line_break)
    print("Using QL policy")
    print(line_break)

    ret, steps, goal = ql_agent.run_eval_episode(
        env, True, 0.01, "readable"
    )
    # ql_agent.run_eval_episode(render=args.render_eval)
