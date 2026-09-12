import random
import json
import math
import os
from statistics import mean, stdev


ACTIONS = ["explore", "observe", "learn"]
FEATURES = ["size", "speed", "energy"]


class Memory:
    def __init__(self, filename="elsadigai_v2_memory.json"):
        self.filename = filename
        self.data = []

    def add(self, state, action, reward, prediction):
        self.data.append({
            "state": state,
            "action": action,
            "reward": reward,
            "prediction": prediction
        })

    def count(self):
        return len(self.data)

    def clear(self):
        self.data = []


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
    """
    يتعلم العلاقة بين الخصائص والأفعال.
    لا يحفظ الحالة كاملة كقاعدة منفردة.
    """

    def __init__(self, memory):
        self.memory = memory
        self.values = {}
        self.counts = {}
        self.learning_rate = 0.15

    def key(self, feature, value, action):
        return f"{feature}:{value}:{action}"

    def get_value(self, feature, value, action):
        key = self.key(feature, value, action)
        return self.values.get(key, 0.0)

    def predict(self, state, action):
        scores = []

        for feature in FEATURES:
            value = state[feature]

            scores.append(
                self.get_value(
                    feature,
                    value,
                    action
                )
            )

        return sum(scores) / len(scores)

    def predictions(self, state):
        return {
            action: self.predict(state, action)
            for action in ACTIONS
        }

    def choose_action(self, state, epsilon=0.0):

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

            old_value = self.values.get(
                key,
                0.0
            )

            self.values[key] = (
                old_value
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


def run_single_experiment(
    seed,
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100
):

    memory = Memory()

    brain = LearningBrain(memory)

    environment = Environment(seed)

    before = evaluate(
        brain,
        environment,
        before_episodes
    )

    training_rewards = train(
        brain,
        environment,
        training_episodes
    )

    after = evaluate(
        brain,
        environment,
        after_episodes
    )

    generalization = evaluate(
        brain,
        environment,
        generalization_episodes
    )

    if before["reward"] != 0:

        reward_improvement = (
            (
                after["reward"]
                - before["reward"]
            )
            / before["reward"]
        ) * 100

    else:
        reward_improvement = 0.0

    accuracy_improvement = (
        after["accuracy"]
        - before["accuracy"]
    )

    return {
        "seed": seed,

        "before_reward":
            before["reward"],

        "before_accuracy":
            before["accuracy"],

        "after_reward":
            after["reward"],

        "after_accuracy":
            after["accuracy"],

        "generalization_reward":
            generalization["reward"],

        "generalization_accuracy":
            generalization["accuracy"],

        "reward_improvement":
            reward_improvement,

        "accuracy_improvement":
            accuracy_improvement,

        "training_reward":
            mean(training_rewards),

        "knowledge_size":
            brain.knowledge_size()
    }


def confidence_interval(values):

    if len(values) < 2:
        return {
            "low": mean(values),
            "high": mean(values)
        }

    avg = mean(values)
    sd = stdev(values)

    se = sd / math.sqrt(len(values))

    margin = 1.96 * se

    return {
        "low": avg - margin,
        "high": avg + margin
    }


def run_learning_benchmark(
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100,
    seed=777
):

    results = []

    experiments = 30

    for i in range(experiments):

        current_seed = seed + i

        result = run_single_experiment(
            current_seed,
            before_episodes,
            training_episodes,
            after_episodes,
            generalization_episodes
        )

        results.append(result)

    before_rewards = [
        r["before_reward"]
        for r in results
    ]

    after_rewards = [
        r["after_reward"]
        for r in results
    ]

    before_accuracy = [
        r["before_accuracy"]
        for r in results
    ]

    after_accuracy = [
        r["after_accuracy"]
        for r in results
    ]

    generalization_accuracy = [
        r["generalization_accuracy"]
        for r in results
    ]

    reward_improvements = [
        r["reward_improvement"]
        for r in results
    ]

    accuracy_improvements = [
        r["accuracy_improvement"]
        for r in results
    ]

    generalization_rewards = [
        r["generalization_reward"]
        for r in results
    ]

    ci_generalization = confidence_interval(
        generalization_accuracy
    )

    successful_generalization = sum(
        1
        for value in generalization_accuracy
        if value >= 50
    )

    report = {

        "version": "ELSADIGAI v2.1",

        "experiments": experiments,

        "before": {
            "reward": mean(before_rewards),
            "accuracy": mean(before_accuracy)
        },

        "after": {
            "reward": mean(after_rewards),
            "accuracy": mean(after_accuracy)
        },

        "generalization": {
            "reward": mean(generalization_rewards),
            "accuracy": mean(generalization_accuracy),

            "minimum_accuracy":
                min(generalization_accuracy),

            "maximum_accuracy":
                max(generalization_accuracy),

            "standard_deviation":
                (
                    stdev(generalization_accuracy)
                    if len(generalization_accuracy) > 1
                    else 0.0
                ),

            "confidence_interval_95": ci_generalization,

            "successful_runs":
                successful_generalization,

            "success_rate":
                (
                    successful_generalization
                    / experiments
                ) * 100
        },

        "improvement": {
            "reward_percent":
                mean(reward_improvements),

            "accuracy_points":
                mean(accuracy_improvements)
        },

        "knowledge": {
            "average_size":
                mean(
                    r["knowledge_size"]
                    for r in results
                )
        },

        "training": {
            "episodes":
                training_episodes,

            "average_reward":
                mean(
                    r["training_reward"]
                    for r in results
                )
        },

        "individual_results": results
    }

    if (
        report["after"]["accuracy"]
        > report["before"]["accuracy"]
        and
        report["generalization"]["accuracy"]
        >= 50
    ):

        report["verdict"] = (
            "تعلم وتعميم قابلان للقياس"
        )

    elif (
        report["after"]["accuracy"]
        > report["before"]["accuracy"]
    ):

        report["verdict"] = (
            "تعلم قابل للقياس "
            "مع تعميم محدود"
        )

    else:

        report["verdict"] = (
            "لم يظهر تعلم واضح"
        )

    return report


def save_benchmark_report(report):

    with open(
        "elsadigai_v2_1_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    report = run_learning_benchmark()

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )