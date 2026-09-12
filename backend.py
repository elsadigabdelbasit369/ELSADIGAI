import random
import json
import os
from statistics import mean


ACTIONS = ["explore", "observe", "learn"]
FEATURES = ["size", "speed", "energy"]


class Memory:
    def __init__(self, filename="elsadigai_v2_memory.json"):
        self.filename = filename
        self.data = []
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []

    def save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(
                self.data,
                f,
                ensure_ascii=False,
                indent=2
            )

    def add(self, state, action, reward, prediction):
        self.data.append({
            "state": state,
            "action": action,
            "reward": reward,
            "prediction": prediction
        })

        if len(self.data) > 2000:
            self.data = self.data[-2000:]

        self.save()

    def clear(self):
        self.data = []
        self.save()

    def count(self):
        return len(self.data)


class Environment:

    def __init__(self, seed=777):
        self.rng = random.Random(seed)

    def create_state(self):
        return {
            "size": self.rng.randint(0, 2),
            "speed": self.rng.randint(0, 2),
            "energy": self.rng.randint(0, 2)
        }

    def optimal_action(self, state):

        score = (
            state["size"]
            + state["speed"]
            - state["energy"]
        )

        if score >= 2:
            return "explore"

        if score <= -1:
            return "observe"

        return "learn"

    def reward(self, state, action):

        correct = self.optimal_action(state)

        if action == correct:
            return 0.8 + self.rng.uniform(0.0, 0.2)

        return self.rng.uniform(0.0, 0.3)


class LearningBrain:

    def __init__(self, memory):

        self.memory = memory

        self.values = {}
        self.counts = {}

        self.learning_rate = 0.15

    def key(self, feature, value, action):
        return f"{feature}:{value}:{action}"

    def get_value(self, feature, value, action):

        key = self.key(
            feature,
            value,
            action
        )

        return self.values.get(key, 0.0)

    def predict(self, state, action):

        scores = []

        for feature in FEATURES:

            value = state[feature]

            score = self.get_value(
                feature,
                value,
                action
            )

            scores.append(score)

        return sum(scores) / len(scores)

    def predictions(self, state):

        return {
            action: self.predict(state, action)
            for action in ACTIONS
        }

    def choose_action(self, state, epsilon=0.10):

        if random.random() < epsilon:
            return random.choice(ACTIONS)

        predictions = self.predictions(state)

        return max(
            predictions,
            key=predictions.get
        )

    def learn(
        self,
        state,
        action,
        reward,
        prediction
    ):

        error = reward - prediction

        for feature in FEATURES:

            value = state[feature]

            key = self.key(
                feature,
                value,
                action
            )

            old = self.values.get(
                key,
                0.0
            )

            self.values[key] = (
                old
                + self.learning_rate * error
            )

            self.counts[key] = (
                self.counts.get(key, 0)
                + 1
            )

        self.memory.add(
            state,
            action,
            reward,
            prediction
        )

    def knowledge_size(self):
        return len(self.values)


def evaluate(
    brain,
    environment,
    episodes=100
):

    total_reward = 0.0
    correct = 0

    for _ in range(episodes):

        state = environment.create_state()

        predictions = brain.predictions(
            state
        )

        action = max(
            predictions,
            key=predictions.get
        )

        optimal = environment.optimal_action(
            state
        )

        reward = environment.reward(
            state,
            action
        )

        total_reward += reward

        if action == optimal:
            correct += 1

    return {
        "reward": total_reward / episodes,
        "accuracy": (
            correct / episodes
        ) * 100
    }


def train(
    brain,
    environment,
    episodes=500
):

    rewards = []

    for _ in range(episodes):

        state = environment.create_state()

        predictions = brain.predictions(
            state
        )

        action = brain.choose_action(
            state,
            epsilon=0.20
        )

        prediction = predictions[action]

        reward = environment.reward(
            state,
            action
        )

        brain.learn(
            state,
            action,
            reward,
            prediction
        )

        rewards.append(reward)

    return rewards


def run_learning_benchmark(
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100,
    seed=777
):

    memory = Memory(
        "elsadigai_v2_benchmark_memory.json"
    )

    memory.clear()

    brain = LearningBrain(memory)

    environment = Environment(seed)

    # BEFORE
    before = evaluate(
        brain,
        environment,
        before_episodes
    )

    # TRAINING
    training_rewards = train(
        brain,
        environment,
        training_episodes
    )

    # AFTER
    after = evaluate(
        brain,
        environment,
        after_episodes
    )

    # GENERALIZATION
    generalization = evaluate(
        brain,
        environment,
        generalization_episodes
    )

    reward_improvement = 0.0

    if before["reward"] != 0:
        reward_improvement = (
            (
                after["reward"]
                - before["reward"]
            )
            / before["reward"]
        ) * 100

    accuracy_improvement = (
        after["accuracy"]
        - before["accuracy"]
    )

    return {
        "before": before,
        "training": {
            "episodes": training_episodes,
            "average_reward": (
                mean(training_rewards)
                if training_rewards
                else 0.0
            )
        },
        "after": after,
        "generalization": generalization,
        "improvement": {
            "reward_percent": reward_improvement,
            "accuracy_points": accuracy_improvement
        },
        "knowledge_size": brain.knowledge_size(),
        "memory_size": memory.count(),
        "verdict": (
            "تعلم قابل للقياس"
            if after["accuracy"]
            > before["accuracy"]
            else "لم يظهر تحسن واضح"
        )
    }


def save_benchmark_report(report):

    with open(
        "elsadigai_v2_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )


def memory_summary():

    memory = Memory(
        "elsadigai_v2_memory.json"
    )

    return {
        "count": memory.count()
    }


if __name__ == "__main__":

    report = run_learning_benchmark()

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )