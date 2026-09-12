import tkinter as tk
from tkinter import messagebox
import random
import threading

from backend import (
    Memory,
    LearningAgent,
    run_learning_benchmark,
    save_benchmark_report
)


# ============================================================
# ELSADIGAI — FRONTEND
# ============================================================

BG = "#10141c"
PANEL = "#181e29"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"


class ELSADIGAIApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "ELSADIGAI"
        )

        self.root.geometry(
            "420x760"
        )

        self.root.configure(
            bg=BG
        )

        self.memory = Memory()

        self.agent = LearningAgent(
            self.memory
        )

        self.build_ui()

    # ========================================================
    # إنشاء الواجهة
    # ========================================================

    def build_ui(self):

        title = tk.Label(
            self.root,
            text="ELSADIGAI",
            font=("Arial", 25, "bold"),
            bg=BG,
            fg=ACCENT
        )

        title.pack(
            pady=(20, 2)
        )

        subtitle = tk.Label(
            self.root,
            text="Learning Intelligence System",
            font=("Arial", 10),
            bg=BG,
            fg=MUTED
        )

        subtitle.pack(
            pady=(0, 15)
        )

        self.content = tk.Frame(
            self.root,
            bg=BG
        )

        self.content.pack(
            fill="both",
            expand=True
        )

        nav = tk.Frame(
            self.root,
            bg=PANEL
        )

        nav.pack(
            fill="x",
            side="bottom"
        )

        self.nav_button(
            nav,
            "الرئيسية",
            self.show_home
        )

        self.nav_button(
            nav,
            "الذاكرة",
            self.show_memory
        )

        self.nav_button(
            nav,
            "الاختبار",
            self.show_test
        )

        self.nav_button(
            nav,
            "التحليل",
            self.show_insights
        )

        self.show_home()

    # ========================================================
    # أزرار التنقل
    # ========================================================

    def nav_button(
        self,
        parent,
        text,
        command
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=PANEL,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground=BG,
            relief="flat",
            font=("Arial", 9, "bold")
        )

        button.pack(
            side="left",
            expand=True,
            fill="both",
            padx=2,
            pady=5
        )

    # ========================================================
    # تنظيف المحتوى
    # ========================================================

    def clear_content(self):

        for widget in self.content.winfo_children():
            widget.destroy()

    # ========================================================
    # بطاقة
    # ========================================================

    def card(
        self,
        parent,
        title,
        value,
        color=TEXT
    ):

        frame = tk.Frame(
            parent,
            bg=PANEL,
            padx=15,
            pady=12
        )

        frame.pack(
            fill="x",
            padx=15,
            pady=6
        )

        tk.Label(
            frame,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=("Arial", 10)
        ).pack(
            anchor="w"
        )

        tk.Label(
            frame,
            text=value,
            bg=PANEL,
            fg=color,
            font=("Arial", 17, "bold")
        ).pack(
            anchor="w"
        )

    # ========================================================
    # الصفحة الرئيسية
    # ========================================================

    def show_home(self):

        self.clear_content()

        tk.Label(
            self.content,
            text="🧠",
            bg=BG,
            fg=ACCENT,
            font=("Arial", 42)
        ).pack(
            pady=(25, 5)
        )

        tk.Label(
            self.content,
            text="نظام ذكاء يتعلم من التجربة",
            bg=BG,
            fg=TEXT,
            font=("Arial", 17, "bold")
        ).pack(
            pady=5
        )

        tk.Label(
            self.content,
            text=(
                "يجرب → يحصل على مكافأة → يتعلم → "
                "يحسن قراره"
            ),
            bg=BG,
            fg=MUTED,
            font=("Arial", 10),
            justify="center"
        ).pack(
            pady=12
        )

        self.card(
            self.content,
            "عدد الخبرات في الذاكرة",
            str(self.memory.count()),
            ACCENT
        )

        tk.Button(
            self.content,
            text="▶ دورة تفكير",
            command=self.thinking_cycle,
            bg=ACCENT,
            fg=BG,
            relief="flat",
            font=("Arial", 13, "bold"),
            padx=20,
            pady=10
        ).pack(
            pady=12
        )

        tk.Button(
            self.content,
            text="🧪 اختبار التعلم الحقيقي",
            command=self.show_test,
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=10
        ).pack(
            pady=5
        )

    # ========================================================
    # دورة التفكير
    # ========================================================

    def thinking_cycle(self):

        states = [
            "state_A",
            "state_B",
            "state_C"
        ]

        state = random.choice(
            states
        )

        action = self.agent.choose_action(
            state,
            epsilon=0.30
        )

        reward = random.uniform(
            0.0,
            1.0
        )

        self.agent.learn(
            state,
            action,
            reward,
            "normal"
        )

        messagebox.showinfo(
            "دورة التفكير",
            (
                f"الحالة: {state}\n\n"
                f"الفعل: {action}\n\n"
                f"المكافأة: {reward:.3f}\n\n"
                "تم حفظ التجربة في الذاكرة."
            )
        )

    # ========================================================
    # صفحة الذاكرة
    # ========================================================

    def show_memory(self):

        self.clear_content()

        tk.Label(
            self.content,
            text="🗂 ذاكرة ELSADIGAI",
            bg=BG,
            fg=TEXT,
            font=("Arial", 19, "bold")
        ).pack(
            pady=20
        )

        self.card(
            self.content,
            "عدد الخبرات",
            str(self.memory.count()),
            ACCENT
        )

        text = tk.Text(
            self.content,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Courier", 9),
            height=20
        )

        text.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        for experience in self.memory.experiences[-30:]:

            text.insert(
                "end",
                (
                    f"الحالة: "
                    f"{experience.get('state')}\n"

                    f"الفعل: "
                    f"{experience.get('action')}\n"

                    f"المكافأة: "
                    f"{experience.get('reward')}\n"

                    f"المصدر: "
                    f"{experience.get('source')}\n"

                    "----------------------\n"
                )
            )

        text.config(
            state="disabled"
        )

        tk.Button(
            self.content,
            text="حذف الذاكرة",
            command=self.clear_memory,
            bg="#3f1d1d",
            fg="#ffaaaa",
            relief="flat",
            font=("Arial", 10, "bold"),
            pady=8
        ).pack(
            fill="x",
            padx=15,
            pady=5
        )

    # ========================================================
    # حذف الذاكرة
    # ========================================================

    def clear_memory(self):

        answer = messagebox.askyesno(
            "تأكيد",
            "هل تريد حذف الذاكرة بالكامل؟"
        )

        if answer:

            self.memory.clear()

            self.agent.reset_learning()

            self.show_memory()

    # ========================================================
    # صفحة اختبار التعلم
    # ========================================================

    def show_test(self):

        self.clear_content()

        tk.Label(
            self.content,
            text="🧪 اختبار التعلم الحقيقي",
            bg=BG,
            fg=TEXT,
            font=("Arial", 19, "bold")
        ).pack(
            pady=20
        )

        tk.Label(
            self.content,
            text=(
                "اختبار مستقل لقياس قدرة النظام على التعلم.\n\n"
                "قبل التعلم\n"
                "↓\n"
                "500 تجربة تدريب\n"
                "↓\n"
                "بعد التعلم\n"
                "↓\n"
                "اختبار التعميم"
            ),
            bg=BG,
            fg=MUTED,
            justify="center",
            font=("Arial", 10)
        ).pack(
            pady=10
        )

        self.progress_label = tk.Label(
            self.content,
            text="جاهز",
            bg=BG,
            fg=ACCENT,
            font=("Arial", 11, "bold")
        )

        self.progress_label.pack(
            pady=10
        )

        self.test_button = tk.Button(
            self.content,
            text="▶ ابدأ الاختبار",
            command=self.start_test,
            bg=ACCENT,
            fg=BG,
            relief="flat",
            font=("Arial", 13, "bold"),
            pady=12
        )

        self.test_button.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.result_text = tk.Text(
            self.content,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Arial", 9),
            height=15
        )

        self.result_text.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        self.result_text.config(
            state="disabled"
        )

    # ========================================================
    # بدء الاختبار
    # ========================================================

    def start_test(self):

        self.test_button.config(
            state="disabled"
        )

        self.result_text.config(
            state="normal"
        )

        self.result_text.delete(
            "1.0",
            "end"
        )

        self.result_text.insert(
            "end",
            "جاري تشغيل الاختبار...\n\n"
        )

        self.result_text.config(
            state="disabled"
        )

        thread = threading.Thread(
            target=self.run_test,
            daemon=True
        )

        thread.start()

    # ========================================================
    # تنفيذ الاختبار
    # ========================================================

    def run_test(self):

        def progress(
            current,
            total,
            source
        ):

            percentage = (
                current /
                total
            ) * 100

            self.root.after(
                0,
                lambda:
                self.progress_label.config(
                    text=(
                        f"{source} : "
                        f"{current}/{total} "
                        f"({percentage:.0f}%)"
                    )
                )
            )

        try:

            report = run_learning_benchmark(
                before_episodes=100,
                training_episodes=500,
                after_episodes=100,
                generalization_episodes=100,
                seed=777,
                progress_callback=progress
            )

            path = save_benchmark_report(
                report
            )

            self.root.after(
                0,
                lambda:
                self.display_report(
                    report,
                    path
                )
            )

        except Exception as error:

            self.root.after(
                0,
                lambda:
                self.show_error(error)
            )

    # ========================================================
    # عرض التقرير
    # ========================================================

    def display_report(
        self,
        report,
        path
    ):

        before = report["before"]

        after = report["after"]

        generalization = \
            report["generalization"]

        text = (

            "================================\n"
            "       ELSADIGAI TEST\n"
            "================================\n\n"

            "قبل التعلم\n"
            f"Reward: "
            f"{before['average_reward']:.4f}\n"

            f"Accuracy: "
            f"{before['best_action_rate'] * 100:.2f}%\n\n"

            "بعد التعلم\n"
            f"Reward: "
            f"{after['average_reward']:.4f}\n"

            f"Accuracy: "
            f"{after['best_action_rate'] * 100:.2f}%\n\n"

            "التعميم\n"
            f"Reward: "
            f"{generalization['average_reward']:.4f}\n"

            f"Accuracy: "
            f"{generalization['best_action_rate'] * 100:.2f}%\n\n"

            "--------------------------------\n"

            f"تحسن المكافأة: "
            f"{report['reward_improvement_pct']:.2f}%\n"

            f"تحسن الدقة: "
            f"{report['accuracy_improvement_pp']:.2f} نقطة\n\n"

            "الحكم:\n"
            f"{report['verdict']}\n\n"

            f"تم حفظ التقرير في:\n"
            f"{path}\n"

            "================================\n"
        )

        self.result_text.config(
            state="normal"
        )

        self.result_text.delete(
            "1.0",
            "end"
        )

        self.result_text.insert(
            "end",
            text
        )

        self.result_text.config(
            state="disabled"
        )

        self.progress_label.config(
            text="✅ اكتمل الاختبار"
        )

        self.test_button.config(
            state="normal"
        )

    # ========================================================
    # عرض الخطأ
    # ========================================================

    def show_error(self, error):

        self.result_text.config(
            state="normal"
        )

        self.result_text.insert(
            "end",
            "\nحدث خطأ:\n"
            + str(error)
        )

        self.result_text.config(
            state="disabled"
        )

        self.progress_label.config(
            text="❌ حدث خطأ"
        )

        self.test_button.config(
            state="normal"
        )

    # ========================================================
    # صفحة التحليل
    # ========================================================

    def show_insights(self):

        self.clear_content()

        tk.Label(
            self.content,
            text="📊 تحليل المعرفة",
            bg=BG,
            fg=TEXT,
            font=("Arial", 19, "bold")
        ).pack(
            pady=20
        )

        self.card(
            self.content,
            "عدد المعارف المكتسبة",
            str(len(self.agent.values)),
            ACCENT
        )

        text = tk.Text(
            self.content,
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            font=("Arial", 9),
            height=20
        )

        text.pack(
            fill="both",
            expand=True,
            padx=15
        )

        states = {}

        for key, value in self.agent.values.items():

            state, action = key.split(
                "|",
                1
            )

            if state not in states:
                states[state] = []

            states[state].append(
                (
                    action,
                    value,
                    self.agent.counts.get(
                        key,
                        0
                    )
                )
            )

        for state, values in states.items():

            values.sort(
                key=lambda x: x[1],
                reverse=True
            )

            text.insert(
                "end",
                f"\nالحالة: {state}\n"
            )

            for action, value, count in values:

                text.insert(
                    "end",
                    (
                        f"  {action}: "
                        f"{value:.3f} "
                        f"({count} تجربة)\n"
                    )
                )

        text.config(
            state="disabled"
        )


# ============================================================
# تشغيل التطبيق
# ============================================================

def launch():

    root = tk.Tk()

    ELSADIGAIApp(root)

    root.mainloop()


if __name__ == "__main__":
    launch()