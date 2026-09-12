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
        source="normal"
    ):

        experience = {
            "state": state,
            "action": action,
            "reward": round(float(reward), 6),
            "source": source,
            "time": datetime.now().isoformat(
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
        return len(self.experiences)


class LearningAgent:

    def __init__(self, memory=None):

        self.memory = memory or Memory()

        self.values = {}
        self.counts = {}

        self.load_knowledge()

    @staticmethod
    def key(state, action):
        return f"{state}|{action}"

    def load_knowledge(self):

        self.values = {}
        self.counts = {}

        for experience in self.memory.experiences:

            state = experience.get("state")
            action = experience.get("action")
            reward = experience.get("reward")

            if (
                state
                and action in ACTIONS
                and reward is not None
            ):
                self._update_value(
                    state,
                    action,
                    float(reward)
                )

    def _update_value(
        self,
        state,
        action,
        reward
    ):

        key = self.key(
            state,
            action
        )

        old_count = self.counts.get(
            key,
            0
        )

        old_value = self.values.get(
            key,
            0.0
        )

        new_count = old_count + 1

        new_value = (
            old_value
            +
            (
                reward - old_value
            )
            /
            new_count
        )

        self.counts[key] = new_count
        self.values[key] = new_value

    def reset_learning(self):

        self.values = {}
        self.counts = {}

    def learn(
        self,
        state,
        action,
        reward,
        source="training"
    ):

        self.memory.add(
            state,
            action,
            reward,
            source
        )

        self._update_value(
            state,
            action,
            float(reward)
        )

    def choose_action(
        self,
        state,
        epsilon=0.20,
        rng=None
    ):

        if rng is None:
            rng = random.Random()

        unknown = [
            action
            for action in ACTIONS
            if self.counts.get(
                self.key(
                    state,
                    action
                ),
                0
            ) == 0
        ]

        # الاستكشاف
        if unknown and rng.random() < epsilon:
            return rng.choice(unknown)

        # الاستكشاف العشوائي
        if rng.random() < epsilon:
            return rng.choice(ACTIONS)

        scores = {
            action: self.values.get(
                self.key(
                    state,
                    action
                ),
                0.0
            )
            for action in ACTIONS
        }

        best_value = max(
            scores.values()
        )

        best_actions = [
            action
            for action, value in scores.items()
            if abs(
                value - best_value
            ) < 1e-12
        ]

        return rng.choice(
            best_actions
        )

    def best_known_action(self, state):

        known = [
            action
            for action in ACTIONS
            if self.counts.get(
                self.key(
                    state,
                    action
                ),
                0
            ) > 0
        ]

        if not known:
            return None

        return max(
            known,
            key=lambda action:
                self.values.get(
                    self.key(
                        state,
                        action
                    ),
                    0.0
                )
        )


class LearningEnvironment:

    """
    بيئة اختبار التعلم.

    الوكيل لا يرى الفعل الأفضل.
    البيئة فقط تستخدمه داخليًا لحساب المكافأة.
    """

    def __init__(self, seed=2026):

        self.rng = random.Random(seed)

        self.optimal = {
            "A": "explore",
            "B": "observe",
            "C": "learn",
            "D": "explore",
            "E": "learn",
            "F": "observe",
            "G": "learn",
            "H": "explore"
        }

    def reward(self, state, action):

        if action == self.optimal[state]:
            mean = 0.90
        else:
            mean = 0.20

        noise = self.rng.uniform(
            -0.10,
            0.10
        )

        reward = mean + noise

        return max(
            0.0,
            min(
                1.0,
                reward
            )
        )

    def random_state(self, states=None):

        if states is None:
            states = list(
                self.optimal.keys()
            )

        return self.rng.choice(states)


def evaluate(
    agent,
    environment,
    episodes,
    states,
    seed,
    epsilon,
    learn,
    source,
    progress_callback=None
):

    rng = random.Random(seed)

    total_reward = 0.0
    correct = 0

    for index in range(episodes):

        state = rng.choice(states)

        action = agent.choose_action(
            state,
            epsilon,
            rng
        )

        reward = environment.reward(
            state,
            action
        )

        total_reward += reward

        if action == environment.optimal[state]:
            correct += 1

        if learn:
            agent.learn(
                state,
                action,
                reward,
                source
            )

        if (
            progress_callback
            and (
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
        "episodes": episodes,
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

    memory = Memory()

    agent = LearningAgent(memory)

    agent.reset_learning()

    training_states = list("ABCDEFGH")

    # ==========================
    # BEFORE
    # ==========================

    before = evaluate(
        agent,
        LearningEnvironment(seed + 1),
        before_episodes,
        training_states,
        seed + 10,
        epsilon=1.0,
        learn=False,
        source="before",
        progress_callback=progress_callback
    )

    # ==========================
    # TRAINING
    # ==========================

    training = evaluate(
        agent,
        LearningEnvironment(seed + 2),
        training_episodes,
        training_states,
        seed + 20,
        epsilon=0.20,
        learn=True,
        source="training",
        progress_callback=progress_callback
    )

    # ==========================
    # AFTER
    # ==========================

    after = evaluate(
        agent,
        LearningEnvironment(seed + 3),
        after_episodes,
        training_states,
        seed + 30,
        epsilon=0.0,
        learn=False,
        source="after",
        progress_callback=progress_callback
    )

    # ==========================
    # GENERALIZATION
    # ==========================

    generalization_environment = \
        LearningEnvironment(seed + 4)

    generalization_environment.optimal.update({
        "I": "observe",
        "J": "learn",
        "K": "explore",
        "L": "observe"
    })

    generalization = evaluate(
        agent,
        generalization_environment,
        generalization_episodes,
        list("IJKL"),
        seed + 40,
        epsilon=0.0,
        learn=False,
        source="generalization",
        progress_callback=progress_callback
    )

    # ==========================
    # المقارنة
    # ==========================

    reward_delta = (
        after["average_reward"]
        -
        before["average_reward"]
    )

    if before["average_reward"] != 0:

        reward_improvement_pct = (
            reward_delta
            /
            abs(
                before["average_reward"]
            )
        ) * 100.0

    else:

        reward_improvement_pct = 0.0

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

        verdict = "🚀 تعلم + تعميم"

    elif learned:

        verdict = "✅ تعلم قابل للقياس"

    elif reward_delta > 0:

        verdict = "⚠️ تحسن محدود"

    else:

        verdict = "❌ لم يظهر تعلم واضح"

    return {

        "before": before,

        "training": training,

        "after": after,

        "generalization":
            generalization,

        "reward_delta":
            reward_delta,

        "reward_improvement_pct":
            reward_improvement_pct,

        "accuracy_improvement_pp":
            accuracy_improvement_pp,

        "learned": learned,

        "generalized": generalized,

        "verdict": verdict,

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


def memory_summary(memory=None):

    memory = memory or Memory()

    return {

        "experiences":
            memory.count(),

        "sources": {

            "normal":
                sum(
                    e.get("source")
                    == "normal"
                    for e in memory.experiences
                ),

            "before":
                sum(
                    e.get("source")
                    == "before"
                    for e in memory.experiences
                ),

            "training":
                sum(
                    e.get("source")
                    == "training"
                    for e in memory.experiences
                )
        }
    }


# ============================================================
# اختبار Backend مستقل
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print("ELSADIGAI v1.1 BACKEND")

    print(
        "بدء اختبار التعلم..."
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
        "التحسن:",
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

    path = save_benchmark_report(
        report
    )

    print()

    print(
        "تم حفظ التقرير:",
        path
    )