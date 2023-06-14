from game import State
from agent import Agent, HumanAgent


if __name__ == "__main__":
    # training
    p1 = Agent("p1")
    p2 = Agent("p2")

    st = State(p1, p2)
    print("training...")
    st.play(50000)
    p1.savePolicy()
    # p2.savePolicy()

    # play with human
    p1 = Agent("computer", exp_rate=0)

    p1.loadPolicy("policy_p1")
    # p2 = Agent("computer", exp_rate=0)
    # p2.loadPolicy("policy_p2")
    p2 = HumanAgent("human")

    st = State(p1, p2)
    st.play2()
    # win_count = st.play(run_eval=True)
    # print(win_count)