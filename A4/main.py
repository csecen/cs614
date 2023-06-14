from game import State
from agent import Agent, HumanAgent


if __name__ == "__main__":
    # training
    p1 = Agent("p1")
    p2 = Agent("p2")

    st = State(p1, p2)
    print("training...")
    st.play(50000)

    # play with human
    # p1 = Agent("computer", exp_rate=0)
    p1.savePolicy()
    # p1.loadPolicy("policy_computer")
    #
    # p2 = HumanAgent("human")
    #
    # st = State(p1, p2)
    # st.play2()