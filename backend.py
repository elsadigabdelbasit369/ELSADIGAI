import random
import json
import math
from statistics import mean, stdev


# ============================================================
# ELSADIGAI v2.2
# Feature Learning + True Held-Out Generalization
# ============================================================

ACTIONS = [
    "explore",
    "observe",
    "learn"
]

FEATURES = [
    "size",
    "speed",
    "energy"
]


# ============================================================
# MEMORY
# ============================================================

class Memory:

    def __init__(self):
        self.data = []

    def add(
        self,
        state,
        action,
        reward,
        prediction
    ):

        self.data.append({
            "state": dict(state),
            "action": action,
            "reward": float(reward),
            "prediction": float(prediction)
        })

    def count(self):
        return len(self.data)

    def clear(self):
        self.data = []


# ============================================================
# ENVIRONMENT
# ============================================================

class Environment:

    def __init__(self, seed=777):

        self.rng = random.Random(seed)

    # --------------------------------------------------------
    # Create random state
    # --------------------------------------------------------

    def create_state(self):

        return {
            "size": self.rng.randint(0, 2),
            "speed": self.rng.randint(0, 2),
            "energy": self.rng.randint(0, 2)
        }

    # --------------------------------------------------------
    # Hidden rule
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    def reward(self, state, action):

        correct_action = self.optimal_action(
            state
        )

        if action == correct_action:

            return (
                0.8
                + self.rng.uniform(
                    0.0,
                    0.2
                )
            )

        return self.rng.uniform(
            0.0,
            0.3
        )


# ============================================================
# STATE GENERATION
# ============================================================

def all_states():

    states = []

    for size in range(3):

        for speed in range(3):

            for energy in range(3):

                states.append({
                    "size": size,
                    "speed": speed,
                    "energy": energy
                })

    return states


def split_states():

    """
    تقسيم الحالات بطريقة تمنع تسرب نفس التركيبة
    من التدريب إلى اختبار التعميم.

    التدريب:
        مجموع الخصائص زوجي

    التعميم:
        مجموع الخصائص فردي
    """

    training_states = []
    generalization_states = []

    for state in all_states():

        total = (
            state["size"]
            + state["speed"]
            + state["energy"]
        )

        if total % 2 == 0:

            training_states.append(state)

        else:

            generalization_states.append(state)

    return (
        training_states,
        generalization_states
    )


# ============================================================
# LEARNING BRAIN
# ============================================================

class LearningBrain:

    def __init__(self, memory):

        self.memory = memory

        # قيمة كل خاصية مع كل فعل
        self.values = {}

        # عدد مرات التعلم
        self.counts = {}

        self.learning_rate = 0.15

    # --------------------------------------------------------
    # Knowledge key
    # --------------------------------------------------------

    def key(
        self,
        feature,
        value,
        action
    ):

        return (
            f"{feature}:"
            f"{value}:"
            f"{action}"
        )

    # --------------------------------------------------------
    # Get knowledge
    # --------------------------------------------------------

    def get_value(
        self,
        feature,
        value,
        action
    ):

        key = self.key(
            feature,
            value,
            action
        )

        return self.values.get(
            key,
            0.0
        )

    # --------------------------------------------------------
    # Predict action reward
    # --------------------------------------------------------

    def predict(
        self,
        state,
        action
    ):

        scores = []

        for feature in FEATURES:

            value = state[feature]

            score = self.get_value(
                feature,
                value,
                action
            )

            scores.append(score)

        if not scores:
            return 0.0

        return (
            sum(scores)
            / len(scores)
        )

    # --------------------------------------------------------
    # Predict all actions
    # --------------------------------------------------------

    def predictions(self, state):

        result = {}

        for action in ACTIONS:

            result[action] = self.predict(
                state,
                action
            )

        return result

    # --------------------------------------------------------
    # Choose action
    # --------------------------------------------------------

    def choose_action(
        self,
        state,
        epsilon=0.20,
        rng=None
    ):

        if rng is None:
            rng = random

        if rng.random() < epsilon:

            return rng.choice(
                ACTIONS
            )

        predictions = self.predictions(
            state
        )

        return max(
            predictions,
            key=predictions.get
        )

    # --------------------------------------------------------
    # Learn
    # --------------------------------------------------------

    def learn(
        self,
        state,
        action,
        reward,
        prediction
    ):

        error = (
            reward
            - prediction
        )

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

            new_value = (
                old_value
                + self.learning_rate
                * error
            )

            self.values[key] = (
                new_value
            )

            self.counts[key] = (
                self.counts.get(
                    key,
                    0
                )
                + 1
            )

        # حفظ التجربة
        self.memory.add(
            state,
            action,
            reward,
            prediction
        )

    # --------------------------------------------------------
    # Knowledge size
    # --------------------------------------------------------

    def knowledge_size(self):

        return len(
            self.values
        )


# ============================================================
# TRAINING
# ============================================================

def train(
    brain,
    environment,
    training_states,
    episodes,
    seed
):

    rng = random.Random(seed)

    rewards = []

    for _ in range(episodes):

        # نختار حالة من حالات التدريب فقط
        state = dict(
            rng.choice(
                training_states
            )
        )

        predictions = (
            brain.predictions(
                state
            )
        )

        action = brain.choose_action(
            state,
            epsilon=0.20,
            rng=rng
        )

        prediction = predictions[
            action
        ]

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

        rewards.append(
            reward
        )

    return rewards


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    brain,
    environment,
    states,
    episodes,
    seed
):

    rng = random.Random(seed)

    total_reward = 0.0

    correct = 0

    for _ in range(episodes):

        state = dict(
            rng.choice(
                states
            )
        )

        predictions = (
            brain.predictions(
                state
            )
        )

        action = max(
            predictions,
            key=predictions.get
        )

        correct_action = (
            environment.optimal_action(
                state
            )
        )

        reward = environment.reward(
            state,
            action
        )

        total_reward += reward

        if action == correct_action:

            correct += 1

    return {

        "reward":
            total_reward / episodes,

        "accuracy":
            (
                correct
                / episodes
            ) * 100
    }


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def confidence_interval(values):

    if not values:

        return {
            "low": 0.0,
            "high": 0.0
        }

    if len(values) < 2:

        value = values[0]

        return {
            "low": value,
            "high": value
        }

    average = mean(values)

    deviation = stdev(values)

    standard_error = (
        deviation
        / math.sqrt(
            len(values)
        )
    )

    margin = (
        1.96
        * standard_error
    )

    return {

        "low":
            average - margin,

        "high":
            average + margin
    }


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_single_experiment(
    seed,
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100
):

    memory = Memory()

    brain = LearningBrain(
        memory
    )

    environment = Environment(
        seed=seed
    )

    (
        training_states,
        generalization_states
    ) = split_states()

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    before = evaluate(
        brain,
        environment,
        training_states,
        before_episodes,
        seed + 1000
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    training_rewards = train(
        brain,
        environment,
        training_states,
        training_episodes,
        seed + 2000
    )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    after = evaluate(
        brain,
        environment,
        training_states,
        after_episodes,
        seed + 3000
    )

    # --------------------------------------------------------
    # TRUE GENERALIZATION
    # --------------------------------------------------------

    generalization = evaluate(
        brain,
        environment,
        generalization_states,
        generalization_episodes,
        seed + 4000
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

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

        "seed":
            seed,

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
            (
                mean(training_rewards)
                if training_rewards
                else 0.0
            ),

        "knowledge_size":
            brain.knowledge_size(),

        "memory_size":
            memory.count(),

        "training_state_count":
            len(training_states),

        "generalization_state_count":
            len(generalization_states)
    }


# ============================================================
# FULL BENCHMARK
# ============================================================

def run_learning_benchmark(
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100,
    seed=777
):

    experiments = 30

    results = []

    # --------------------------------------------------------
    # Run 30 independent experiments
    # --------------------------------------------------------

    for i in range(
        experiments
    ):

        current_seed = (
            seed + i
        )

        result = run_single_experiment(
            current_seed,
            before_episodes,
            training_episodes,
            after_episodes,
            generalization_episodes
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Collect metrics
    # --------------------------------------------------------

    before_rewards = [
        r["before_reward"]
        for r in results
    ]

    before_accuracy = [
        r["before_accuracy"]
        for r in results
    ]

    after_rewards = [
        r["after_reward"]
        for r in results
    ]

    after_accuracy = [
        r["after_accuracy"]
        for r in results
    ]

    generalization_rewards = [
        r["generalization_reward"]
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

    knowledge_sizes = [
        r["knowledge_size"]
        for r in results
    ]

    memory_sizes = [
        r["memory_size"]
        for r in results
    ]

    training_rewards = [
        r["training_reward"]
        for r in results
    ]

    # --------------------------------------------------------
    # Generalization statistics
    # --------------------------------------------------------

    generalization_average = (
        mean(
            generalization_accuracy
        )
    )

    generalization_sd = (
        stdev(
            generalization_accuracy
        )
        if len(
            generalization_accuracy
        ) > 1
        else 0.0
    )

    generalization_ci = (
        confidence_interval(
            generalization_accuracy
        )
    )

    successful_runs = sum(

        1

        for value
        in generalization_accuracy

        if value >= 50.0
    )

    # --------------------------------------------------------
    # Training / generalization states
    # --------------------------------------------------------

    (
        training_states,
        generalization_states
    ) = split_states()

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    average_before_accuracy = mean(
        before_accuracy
    )

    average_after_accuracy = mean(
        after_accuracy
    )

    if (
        average_after_accuracy
        > average_before_accuracy
        and
        generalization_average
        >= 50.0
    ):

        verdict = (
            "تعلم وتعميم قابلان للقياس"
        )

    elif (
        average_after_accuracy
        > average_before_accuracy
    ):

        verdict = (
            "تعلم قابل للقياس "
            "مع تعميم محدود"
        )

    else:

        verdict = (
            "لم يظهر تعلم واضح"
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {

        "version":
            "ELSADIGAI v2.2",

        "experiments":
            experiments,

        "before": {

            "reward":
                mean(
                    before_rewards
                ),

            "accuracy":
                mean(
                    before_accuracy
                )
        },

        "after": {

            "reward":
                mean(
                    after_rewards
                ),

            "accuracy":
                mean(
                    after_accuracy
                )
        },

        "generalization": {

            "reward":
                mean(
                    generalization_rewards
                ),

            "accuracy":
                generalization_average,

            "standard_deviation":
                generalization_sd,

            "minimum_accuracy":
                min(
                    generalization_accuracy
                ),

            "maximum_accuracy":
                max(
                    generalization_accuracy
                ),

            "confidence_interval_95":
                generalization_ci,

            "successful_runs":
                successful_runs,

            "success_rate":
                (
                    successful_runs
                    / experiments
                ) * 100
        },

        "improvement": {

            "reward_percent":
                mean(
                    reward_improvements
                ),

            "accuracy_points":
                mean(
                    accuracy_improvements
                )
        },

        "training": {

            "episodes":
                training_episodes,

            "average_reward":
                mean(
                    training_rewards
                )
        },

        "knowledge": {

            "average_size":
                mean(
                    knowledge_sizes
                )
        },

        "memory": {

            "average_size":
                mean(
                    memory_sizes
                )
        },

        "state_space": {

            "total_states":
                len(
                    all_states()
                ),

            "training_states":
                len(
                    training_states
                ),

            "generalization_states":
                len(
                    generalization_states
                )
        },

        "verdict":
            verdict,

        "individual_results":
            results
    }

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_benchmark_report(
    report
):

    with open(
        "elsadigai_v2_2_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    report = run_learning_benchmark()

    save_benchmark_report(
        report
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )