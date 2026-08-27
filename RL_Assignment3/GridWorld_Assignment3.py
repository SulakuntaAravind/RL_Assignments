"""
Lab Assignment 3: GridWorld – Policy Evaluation and Value Iteration

Simple 1 x 5 GridWorld used for this assignment:
S1 -> S2 -> S3 -> S4 -> S5 (Goal)

Assumptions (because the supplied assignment sheet does not specify a
particular grid layout/reward table):
- States: S1, S2, S3, S4, S5
- Start state: S1
- Goal/terminal state: S5
- Actions: Left, Right
- Moving to a non-goal state: reward = -1
- Moving into S5: reward = +10
- If an action tries to move outside the grid, the agent stays in the
  same state and receives -1
- Discount factor gamma = 0.9
- Given policy for Policy Evaluation: always choose Right
"""

GAMMA = 0.9
STATES = [1, 2, 3, 4, 5]
START_STATE = 1
GOAL_STATE = 5
ACTIONS = ["L", "R"]


def step(state, action):
    """Return (next_state, reward, done)."""
    if state == GOAL_STATE:
        return state, 0, True

    if action == "R":
        next_state = min(state + 1, GOAL_STATE)
    elif action == "L":
        next_state = max(state - 1, START_STATE)
    else:
        raise ValueError("Action must be 'L' or 'R'.")

    reward = 10 if next_state == GOAL_STATE else -1
    done = next_state == GOAL_STATE
    return next_state, reward, done


# Given policy: always move right.
def given_policy(state):
    return "R"


def policy_evaluation(policy, gamma=GAMMA, theta=1e-8, max_iterations=100):
    """Evaluate a fixed policy using iterative Bellman updates."""
    V = {s: 0.0 for s in STATES}
    history = []

    for iteration in range(1, max_iterations + 1):
        new_V = V.copy()

        for s in STATES:
            if s == GOAL_STATE:
                new_V[s] = 0.0
                continue

            action = policy(s)
            next_state, reward, _ = step(s, action)
            new_V[s] = reward + gamma * V[next_state]

        delta = max(abs(new_V[s] - V[s]) for s in STATES)
        history.append((iteration, delta, new_V.copy()))
        V = new_V

        if delta < theta:
            break

    return V, history


def value_iteration(gamma=GAMMA, theta=1e-8, max_iterations=100):
    """Find the optimal value function and greedy optimal policy."""
    V = {s: 0.0 for s in STATES}
    history = []

    for iteration in range(1, max_iterations + 1):
        new_V = V.copy()

        for s in STATES:
            if s == GOAL_STATE:
                new_V[s] = 0.0
                continue

            action_values = []
            for action in ACTIONS:
                next_state, reward, _ = step(s, action)
                q = reward + gamma * V[next_state]
                action_values.append(q)

            new_V[s] = max(action_values)

        delta = max(abs(new_V[s] - V[s]) for s in STATES)
        history.append((iteration, delta, new_V.copy()))
        V = new_V

        if delta < theta:
            break

    optimal_policy = {}
    for s in STATES:
        if s == GOAL_STATE:
            optimal_policy[s] = "-"
            continue

        q_values = {}
        for action in ACTIONS:
            next_state, reward, _ = step(s, action)
            q_values[action] = reward + gamma * V[next_state]

        optimal_policy[s] = max(q_values, key=q_values.get)

    return V, optimal_policy, history


def run_agent(policy, seed=None, max_steps=50):
    """Run one episode and return path, steps, reward and goal status."""
    import random

    rng = random.Random(seed) if seed is not None else random

    state = START_STATE
    path = [state]
    total_reward = 0

    for _ in range(max_steps):
        action = policy(state)
        if action not in ACTIONS:
            raise ValueError("Policy returned an invalid action.")

        next_state, reward, done = step(state, action)
        total_reward += reward
        state = next_state
        path.append(state)

        if done:
            return path, len(path) - 1, total_reward, True

    return path, max_steps, total_reward, False


def random_policy(state):
    import random
    return random.choice(ACTIONS)


def show_value_table(history, first_n=5):
    print("\nPolicy Evaluation Observation Table")
    print("-" * 70)
    print(f"{'Iteration':<12}{'Max Change':<15}{'S1':<10}{'S2':<10}{'S3':<10}{'S4':<10}{'Status'}")
    print("-" * 70)

    for iteration, delta, values in history[:first_n]:
        status = "Converged" if delta < 1e-8 else "Not converged"
        print(
            f"{iteration:<12}{delta:<15.2f}"
            f"{values[1]:<10.2f}{values[2]:<10.2f}"
            f"{values[3]:<10.2f}{values[4]:<10.2f}{status}"
        )


def show_policy(policy, values):
    print("\nOptimal Policy / Value Table")
    print("-" * 45)
    print(f"{'State':<10}{'Best Action':<15}{'Value':<10}")
    print("-" * 45)
    for s in STATES:
        print(f"S{s:<9}{policy[s]:<15}{values[s]:.2f}")


def main():
    print("=" * 70)
    print("GRIDWORLD: POLICY EVALUATION AND VALUE ITERATION")
    print("=" * 70)

    print("\nGridWorld:")
    print("[S1] -- [S2] -- [S3] -- [S4] -- [S5: GOAL]")
    print("Actions: L = Left, R = Right")
    print("Rewards: -1 per normal move, +10 on entering S5")

    # 1. Policy Evaluation
    evaluated_values, eval_history = policy_evaluation(given_policy)
    show_value_table(eval_history)

    print("\nFinal evaluated values:")
    for s in STATES:
        print(f"V(S{s}) = {evaluated_values[s]:.2f}")

    # 2. Value Iteration
    optimal_values, optimal_policy, vi_history = value_iteration()

    print("\nValue Iteration converged in", len(vi_history), "iterations.")
    show_policy(optimal_policy, optimal_values)

    print("\nOptimal policy arrows:")
    print("S1 -> S2 -> S3 -> S4 -> S5")
    print("↑ equivalent action at each state:  R    R    R    R    GOAL")

    # 3. Run random, evaluated and optimal policies.
    # Seed 0 makes the random-policy result reproducible for the report.
    random_path, random_steps, random_reward, random_goal = run_agent(
        random_policy, seed=0
    )
    eval_path, eval_steps, eval_reward, eval_goal = run_agent(given_policy)
    opt_path, opt_steps, opt_reward, opt_goal = run_agent(
        lambda s: optimal_policy[s]
    )

    print("\nPolicy Performance")
    print("-" * 80)
    print(f"{'Policy':<20}{'Path':<35}{'Steps':<8}{'Reward':<10}{'Goal'}")
    print("-" * 80)
    print(
        f"{'Random Policy':<20}"
        f"{' -> '.join('S'+str(x) for x in random_path):<35}"
        f"{random_steps:<8}{random_reward:<10}{random_goal}"
    )
    print(
        f"{'Evaluated Policy':<20}"
        f"{' -> '.join('S'+str(x) for x in eval_path):<35}"
        f"{eval_steps:<8}{eval_reward:<10}{eval_goal}"
    )
    print(
        f"{'Optimal Policy':<20}"
        f"{' -> '.join('S'+str(x) for x in opt_path):<35}"
        f"{opt_steps:<8}{opt_reward:<10}{opt_goal}"
    )

    print("\nAnalysis:")
    print("1. A state is the current position of the agent in GridWorld.")
    print("2. The agent can move Left or Right.")
    print("3. Policy Evaluation calculates how good each state is for a fixed policy.")
    print("4. Value Iteration directly searches for the best value and best action.")
    print("5. Bellman optimality: V*(s) = max_a [R(s,a) + gamma * V*(s')].")
    print("6. The optimal policy reaches the goal in the fewest steps.")
    print("7. It is better than random because it consistently chooses the action")
    print("   with the highest expected return.")
    print("8. Obstacles can block actions and force the optimal path to go around them.")


if __name__ == "__main__":
    main()