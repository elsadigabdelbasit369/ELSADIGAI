import json
import os
import random
from datetime import datetime


MEMORY_FILE = "elsadigai_memory.json"

ACTIONS = [
    "explore",
    "observe",
    "learn"
]

FEATURE_NAMES = [
    "size",
    "speed",
    "energy"
]


class Memory:

    def __init__(self, filename=MEMORY_FILE):
        self.filename = filename
        self.experiences = []
        self.load()

    def load(self):

        try:

            if os.path.exists(self.filename):

                with open(
                    self.filename,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                if isinstance(data, list):

                    self.experiences = data[-1000:]

        except Exception:

            self.experiences = []

    def save(self):

        try:

            with open(
                self.filename,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.experiences[-1000:],
                    file,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception:

            pass

    def add(
        self,
        state,
        action,
        reward,
        source="normal",
        features=None
    ):

        experience = {

            "state": state,

            "action": action,

            "reward":
                round(
                    float(reward),
                    6
                ),

            "source": source,

            "features":
                features,

            "time":
                datetime.now().isoformat(
                    timespec="seconds"
                )
        }

        self.experiences.append(
            experience
        )

        self.experiences = \
            self.experiences[-1000:]

        self.save()

    def clear(self):

        self.experiences = []

        self.save()

    def count(self):

        return len(
            self.experiences
        )


class FeatureLearningAgent:

    """
    وكيل يتعلم من خصائص الحالة.

    لا يخزن فقط:
        الحالة -> الفعل

    بل يتعلم:
        الخصائص -> قيمة الفعل

    ثم يجمع هذه المعرفة لاتخاذ قرار
    في حالة جديدة.
    """

    def __init__(self, memory=None):

        self.memory = memory or Memory()

        self.feature_values = {}

        self.feature_counts = {}

        self.load_knowledge()

    def feature_key(
        self,
        feature_name,
        feature_value,
        action
    ):

        return (
            f"{feature_name}|"
            f"{feature_value}|"
            f"{action}"
        )

    def load_knowledge(self):

        self.feature_values = {}

        self.feature_counts = {}

        for experience in self.memory.experiences:

            features = experience.get(
                "features"
            )

            action = experience.get(
                "action"
            )

            reward = experience.get(
                "reward"
            )

            if (
                isinstance(features, dict)
                and action in ACTIONS
                and reward is not None
            ):

                self._update_features(
                    features,
                    action,
                    float(reward)
                )

    def _update_one(
        self,
        feature_name,
        feature_value,
        action,
        reward
    ):

        key = self.feature_key(
            feature_name,
            feature_value,
            action
        )

        old_count = self.feature_counts.get(
            key,
            0
        )

        old_value = self.feature_values.get(
            key,
            0.0
        )

        new_count = old_count + 1

        new_value = (
            old_value
            +
            (
                reward
                -
                old_value
            )
            /
            new_count
        )

        self.feature_counts[key] = \
            new_count

        self.feature_values[key] = \
            new_value

    def _update_features(
        self,
        features,
        action,
        reward
    ):

        for feature_name in FEATURE_NAMES:

            if feature_name not in features:
                continue

            self._update_one(
                feature_name,
                features[feature_name],
                action,
                reward
            )

    def learn(
        self,
        state,
        features,
        action,
        reward,
        source="training"
    ):

        self.memory.add(
            state,
            action,
            reward,
            source,
            features
        )

        self._update_features(
            features,
            action,
            float(reward)
        )

    def reset_learning(self):

        self.feature_values = {}

        self.feature_counts = {}

    def action_score(
        self,
        features,
        action
    ):

        scores = []

        for feature_name in FEATURE_NAMES:

            if feature_name not in features:
                continue

            key = self.feature_key(
                feature_name,
                features[feature_name],
                action
            )

            count = self.feature_counts.get(
                key,
                0
            )

            if count > 0:

                scores.append(
                    self.feature_values[key]
                )

        if not scores:

            return 0.0

        return sum(scores) / len(scores)

    def choose_action(
        self,
        features,
        epsilon=0.20,
        rng=None
    ):

        if rng is None:

            rng = random.Random()

        if rng.random() < epsilon:

            return rng.choice(
                ACTIONS
            )

        scores = {

            action:
                self.action_score(
                    features,
                    action
                )

            for action in ACTIONS

        }

        best_value = max(
            scores.values()
        )

        best_actions = [

            action

            for action, value
            in scores.items()

            if abs(
                value
                -
                best_value
            ) < 1e-12

        ]

        return rng.choice(
            best_actions
        )


class StructuredEnvironment:

    """
    بيئة لها قاعدة مخفية.

    الوكيل لا يحصل على القاعدة.

    القاعدة تعتمد على:
        size
        speed
        energy
    """

    def __init__(self, seed=2026):

        self.rng = random.Random(
            seed
        )

    def state_name(
        self,
        features
    ):

        return (
            f"S={features['size']};"
            f"V={features['speed']};"
            f"E={features['energy']}"
        )

    def optimal_action(
        self,
        features
    ):

        size = features["size"]

        speed = features["speed"]

        energy = features["energy"]

        score = (
            size
            +
            speed
            -
            energy
        )

        if score >= 2:

            return "explore"

        if score <= -1:

            return "observe"

        return "learn"

    def reward(
        self,
        features,
        action
    ):

        correct = self.optimal_action(
            features
        )

        if action == correct:

            mean = 0.90

        else:

            mean = 0.20

        noise = self.rng.uniform(
            -0.10,
            0.10
        )

        return max(
            0.0,
            min(
                1.0,
                mean + noise
            )
        )


def all_feature_states():

    states = []

    for size in [0, 1, 2]:

        for speed in [0, 1, 2]:

            for energy in [0, 1, 2]:

                states.append({

                    "size": size,

                    "speed": speed,

                    "energy": energy

                })

    return states


def split_feature_states():

    all_states = \
        all_feature_states()

    training = []

    generalization = []

    for features in all_states:

        total = (
            features["size"]
            +
            features["speed"]
            +
            features["energy"]
        )

        if total % 2 == 0:

            training.append(
                features
            )

        else:

            generalization.append(
                features
            )

    return (
        training,
        generalization
    )


def evaluate(
    agent,
    environment,
    episodes,
    feature_pool,
    seed,
    epsilon,
    learn,
    source,
    progress_callback=None
):

    rng = random.Random(
        seed
    )

    total_reward = 0.0

    correct = 0

    for index in range(
        episodes
    ):

        features = dict(
            rng.choice(
                feature_pool
            )
        )

        state = environment.state_name(
            features
        )

        action = agent.choose_action(
            features,
            epsilon,
            rng
        )

        reward = environment.reward(
            features,
            action
        )

        total_reward += reward

        optimal = environment.optimal_action(
            features
        )

        if action == optimal:

            correct += 1

        if learn:

            agent.learn(
                state,
                features,
                action,
                reward,
                source
            )

        if (
            progress_callback
            and
            (
                index == episodes - 1
                or index % 10 == 0
            )
        ):

            progress_callback(
                index + 1,
                episodes,
                source
            )

    return {

        "episodes":
            episodes,

        "average_reward":
            total_reward / episodes,

        "best_action_rate":
            correct / episodes,

        "total_reward":
            total_reward
    }


def run_learning_benchmark(
    before_episodes=100,
    training_episodes=500,
    after_episodes=100,
    generalization_episodes=100,
    seed=777,
    progress_callback=None
):

    # ========================================================
    # ذاكرة مستقلة للاختبار
    # ========================================================

    memory = Memory(
        filename=(
            "elsadigai_benchmark_memory.json"
        )
    )

    memory.clear()

    agent = FeatureLearningAgent(
        memory
    )

    agent.reset_learning()

    environment = StructuredEnvironment(
        seed + 1
    )

    (
        training_features,
        generalization_features
    ) = split_feature_states()

    # ========================================================
    # BEFORE
    # ========================================================

    before = evaluate(
        agent,
        environment,
        before_episodes,
        training_features,
        seed + 10,
        epsilon=1.0,
        learn=False,
        source="before",
        progress_callback=progress_callback
    )

    # ========================================================
    # TRAINING
    # ========================================================

    training = evaluate(
        agent,
        environment,
        training_episodes,
        training_features,
        seed + 20,
        epsilon=0.20,
        learn=True,
        source="training",
        progress_callback=progress_callback
    )

    # ========================================================
    # AFTER
    # ========================================================

    after = evaluate(
        agent,
        environment,
        after_episodes,
        training_features,
        seed + 30,
        epsilon=0.0,
        learn=False,
        source="after",
        progress_callback=progress_callback
    )

    # ========================================================
    # GENERALIZATION
    # ========================================================

    generalization = evaluate(
        agent,
        environment,
        generalization_episodes,
        generalization_features,
        seed + 40,
        epsilon=0.0,
        learn=False,
        source="generalization",
        progress_callback=progress_callback
    )

    # ========================================================
    # المقارنة
    # ========================================================

    reward_delta = (
        after["average_reward"]
        -
        before["average_reward"]
    )

    reward_improvement_pct = (

        reward_delta
        /
        max(
            abs(
                before["average_reward"]
            ),
            1e-12
        )

    ) * 100.0

    accuracy_improvement_pp = (

        after["best_action_rate"]
        -
        before["best_action_rate"]

    ) * 100.0

    learned = (

        reward_delta > 0
        and
        after["best_action_rate"]
        >
        before["best_action_rate"]

    )

    generalized = (

        generalization[
            "best_action_rate"
        ]
        >=
        0.50

    )

    if learned and generalized:

        verdict = (
            "🚀 تعلم + تعميم حقيقي"
        )

    elif learned:

        verdict = (
            "✅ تعلم قابل للقياس"
        )

    elif generalized:

        verdict = (
            "🌐 تعميم بدون تحسن واضح"
        )

    else:

        verdict = (
            "❌ لم يظهر تعلم أو تعميم واضح"
        )

    return {

        "before":
            before,

        "training":
            training,

        "after":
            after,

        "generalization":
            generalization,

        "reward_delta":
            reward_delta,

        "reward_improvement_pct":
            reward_improvement_pct,

        "accuracy_improvement_pp":
            accuracy_improvement_pp,

        "learned":
            learned,

        "generalized":
            generalized,

        "training_states":
            len(training_features),

        "generalization_states":
            len(generalization_features),

        "verdict":
            verdict,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            )
    }


def save_benchmark_report(
    report,
    directory="."
):

    directory = os.path.abspath(
        directory
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    filename = (
        "elsadigai_learning_test_"
        +
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        ".json"
    )

    path = os.path.join(
        directory,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    return path


def memory_summary(
    memory=None
):

    memory = memory or Memory()

    return {

        "experiences":
            memory.count()
    }


if __name__ == "__main__":

    print("=" * 60)

    print(
        "ELSADIGAI v1.3"
    )

    print(
        "FEATURE LEARNING + GENERALIZATION"
    )

    print("=" * 60)

    report = run_learning_benchmark()

    print()

    print("قبل التعلم:")

    print(
        "Reward =",
        round(
            report["before"][
                "average_reward"
            ],
            4
        )
    )

    print(
        "Accuracy =",
        round(
            report["before"][
                "best_action_rate"
            ] * 100,
            2
        ),
        "%"
    )

    print()

    print("بعد التعلم:")

    print(
        "Reward =",
        round(
            report["after"][
                "average_reward"
            ],
            4
        )
    )

    print(
        "Accuracy =",
        round(
            report["after"][
                "best_action_rate"
            ] * 100,
            2
        ),
        "%"
    )

    print()

    print("التعميم:")

    print(
        "Reward =",
        round(
            report["generalization"][
                "average_reward"
            ],
            4
        )
    )

    print(
        "Accuracy =",
        round(
            report["generalization"][
                "best_action_rate"
            ] * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "حالات التدريب:",
        report["training_states"]
    )

    print(
        "حالات التعميم:",
        report["generalization_states"]
    )

    print()

    print(
        "تحسن المكافأة:",
        round(
            report[
                "reward_improvement_pct"
            ],
            2
        ),
        "%"
    )

    print(
        "تحسن الدقة:",
        round(
            report[
                "accuracy_improvement_pp"
            ],
            2
        ),
        "نقطة"
    )

    print()

    print(
        "الحكم:",
        report["verdict"]
    )

    print("=" * 60)