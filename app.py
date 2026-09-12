from flask import Flask, render_template_string, jsonify
from backend import run_learning_benchmark


app = Flask(__name__)


HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>ELSADIGAI</title>

    <style>

        body {
            margin: 0;
            background: #10141c;
            color: #f1f5f9;
            font-family: Arial, sans-serif;
        }

        .container {
            max-width: 700px;
            margin: auto;
            padding: 30px 18px;
        }

        h1 {
            text-align: center;
            color: #38bdf8;
            font-size: 42px;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 30px;
        }

        .card {
            background: #181e29;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 15px;
        }

        .card h2 {
            margin-top: 0;
            color: #38bdf8;
        }

        .value {
            font-size: 28px;
            font-weight: bold;
        }

        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: #38bdf8;
            color: #10141c;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
        }

        button:hover {
            opacity: 0.85;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status {
            text-align: center;
            color: #94a3b8;
            margin: 20px 0;
        }

        .success {
            color: #22c55e;
        }

        .warning {
            color: #f59e0b;
        }

        .error {
            color: #ef4444;
        }

        .result {
            display: none;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #293241;
            padding: 10px 0;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .verdict {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 15px;
        }

        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 30px;
            font-size: 13px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>ELSADIGAI</h1>

    <div class="subtitle">
        Learning Intelligence System
    </div>


    <div class="card">

        <h2>🧠 النظام</h2>

        <p>
            نظام تجريبي لدراسة التعلم من التجربة
            والذاكرة وتحسين اتخاذ القرار.
        </p>

        <p>
            تجربة → مكافأة → ذاكرة → تعلم → قرار أفضل
        </p>

    </div>


    <div class="card">

        <h2>🧪 اختبار التعلم الحقيقي</h2>

        <p>
            سيقارن النظام بين الأداء قبل التعلم
            وبعد التدريب، ثم يجري اختبار تعميم.
        </p>

        <button
            id="runButton"
            onclick="runTest()">

            ▶ تشغيل الاختبار

        </button>

        <div
            id="status"
            class="status">

            جاهز للاختبار

        </div>

    </div>


    <div
        id="result"
        class="result">


        <div class="card">

            <h2>📊 قبل التعلم</h2>

            <div class="metric">

                <span>متوسط المكافأة</span>

                <span
                    id="beforeReward"
                    class="value">
                </span>

            </div>

            <div class="metric">

                <span>دقة اختيار الفعل</span>

                <span
                    id="beforeAccuracy">
                </span>

            </div>

        </div>


        <div class="card">

            <h2>📈 بعد التعلم</h2>

            <div class="metric">

                <span>متوسط المكافأة</span>

                <span
                    id="afterReward"
                    class="value">
                </span>

            </div>

            <div class="metric">

                <span>دقة اختيار الفعل</span>

                <span
                    id="afterAccuracy">
                </span>

            </div>

        </div>


        <div class="card">

            <h2>🌐 اختبار التعميم</h2>

            <div class="metric">

                <span>متوسط المكافأة</span>

                <span
                    id="generalizationReward"
                    class="value">
                </span>

            </div>

            <div class="metric">

                <span>دقة التعميم</span>

                <span
                    id="generalizationAccuracy">
                </span>

            </div>

        </div>


        <div class="card">

            <h2>📐 التحسن</h2>

            <div class="metric">

                <span>تحسن المكافأة</span>

                <span
                    id="rewardImprovement">
                </span>

            </div>

            <div class="metric">

                <span>تحسن الدقة</span>

                <span
                    id="accuracyImprovement">
                </span>

            </div>

        </div>


        <div class="card">

            <h2>🔬 الحكم التجريبي</h2>

            <div
                id="verdict"
                class="verdict">
            </div>

        </div>


    </div>


    <div class="footer">

        ELSADIGAI — Experimental Research Prototype

    </div>

</div>


<script>

async function runTest() {

    const button =
        document.getElementById("runButton");

    const status =
        document.getElementById("status");

    const result =
        document.getElementById("result");


    button.disabled = true;

    result.style.display = "none";

    status.className = "status";

    status.innerText =
        "⏳ جاري تشغيل الاختبار...";


    try {

        const response =
            await fetch("/api/test");


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "حدث خطأ"
            );

        }


        document.getElementById(
            "beforeReward"
        ).innerText =
            data.before.average_reward.toFixed(4);


        document.getElementById(
            "beforeAccuracy"
        ).innerText =
            (
                data.before.best_action_rate * 100
            ).toFixed(2) + "%";


        document.getElementById(
            "afterReward"
        ).innerText =
            data.after.average_reward.toFixed(4);


        document.getElementById(
            "afterAccuracy"
        ).innerText =
            (
                data.after.best_action_rate * 100
            ).toFixed(2) + "%";


        document.getElementById(
            "generalizationReward"
        ).innerText =
            data.generalization.average_reward
                .toFixed(4);


        document.getElementById(
            "generalizationAccuracy"
        ).innerText =
            (
                data.generalization.best_action_rate
                * 100
            ).toFixed(2) + "%";


        document.getElementById(
            "rewardImprovement"
        ).innerText =
            data.reward_improvement_pct
                .toFixed(2) + "%";


        document.getElementById(
            "accuracyImprovement"
        ).innerText =
            data.accuracy_improvement_pp
                .toFixed(2) + " نقطة";


        document.getElementById(
            "verdict"
        ).innerText =
            data.verdict;


        result.style.display = "block";

        status.className =
            "status success";

        status.innerText =
            "✅ اكتمل الاختبار";


    } catch (error) {

        status.className =
            "status error";

        status.innerText =
            "❌ " + error.message;

    }


    button.disabled = false;

}

</script>


</body>

</html>
"""


@app.route("/")
def home():

    return render_template_string(HTML)


@app.route("/api/test")
def test():

    try:

        report = run_learning_benchmark(
            before_episodes=100,
            training_episodes=500,
            after_episodes=100,
            generalization_episodes=100,
            seed=777
        )

        return jsonify(report)

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    import os

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )