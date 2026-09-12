from flask import Flask, jsonify, render_template_string
from backend import run_learning_benchmark

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>ELSADIGAI</title>

<style>

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #0b1020;
    color: white;
}

.container {
    max-width: 900px;
    margin: auto;
    padding: 25px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.header p {
    color: #aab3c5;
}

button {
    display: block;
    margin: 25px auto;
    padding: 15px 30px;
    border: none;
    border-radius: 12px;
    background: #2563eb;
    color: white;
    font-size: 18px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

button:disabled {
    opacity: 0.6;
}

.status {
    text-align: center;
    margin: 15px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 15px;
}

.card {
    background: #151c32;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,.2);
}

.card h2 {
    margin-top: 0;
}

.value {
    font-size: 30px;
    font-weight: bold;
    margin: 10px 0;
}

.small {
    color: #9ca3af;
}

.verdict {
    margin-top: 25px;
    padding: 20px;
    background: #13233b;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
}

.error {
    background: #451a1a;
    padding: 15px;
    border-radius: 10px;
    color: #fecaca;
}

</style>
</head>

<body>

<div class="container">

<div class="header">

<h1>ELSADIGAI</h1>

<p>Learning Intelligence System</p>

<p>
🧠 نظام تجريبي لدراسة التعلم من التجربة والذاكرة
وتحسين اتخاذ القرار.
</p>

</div>

<button id="runButton" onclick="runTest()">
▶ تشغيل الاختبار
</button>

<div id="status" class="status"></div>

<div id="results"></div>

</div>

<script>

function number(value) {

    if (value === undefined || value === null) {
        return 0;
    }

    return Number(value);
}


async function runTest() {

    const button = document.getElementById("runButton");
    const status = document.getElementById("status");
    const results = document.getElementById("results");

    button.disabled = true;

    status.innerHTML = "⏳ جاري تشغيل التجربة...";
    results.innerHTML = "";

    try {

        const response = await fetch("/api/test");

        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }

        const data = await response.json();

        console.log("ELSADIGAI RESULT:", data);

        const before = data.before || {};
        const after = data.after || {};
        const generalization = data.generalization || {};
        const improvement = data.improvement || {};
        const training = data.training || {};

        results.innerHTML = `

        <h2>📊 قبل التعلم</h2>

        <div class="grid">

            <div class="card">

                <h2>متوسط المكافأة</h2>

                <div class="value">
                    ${number(before.reward).toFixed(4)}
                </div>

            </div>

            <div class="card">

                <h2>دقة اختيار الفعل</h2>

                <div class="value">
                    ${number(before.accuracy).toFixed(2)}%
                </div>

            </div>

        </div>


        <h2>📈 بعد التعلم</h2>

        <div class="grid">

            <div class="card">

                <h2>متوسط المكافأة</h2>

                <div class="value">
                    ${number(after.reward).toFixed(4)}
                </div>

            </div>

            <div class="card">

                <h2>دقة اختيار الفعل</h2>

                <div class="value">
                    ${number(after.accuracy).toFixed(2)}%
                </div>

            </div>

        </div>


        <h2>🌐 اختبار التعميم</h2>

        <div class="grid">

            <div class="card">

                <h2>متوسط المكافأة</h2>

                <div class="value">
                    ${number(generalization.reward).toFixed(4)}
                </div>

            </div>

            <div class="card">

                <h2>دقة التعميم</h2>

                <div class="value">
                    ${number(generalization.accuracy).toFixed(2)}%
                </div>

            </div>

        </div>


        <h2>📐 التحسن</h2>

        <div class="grid">

            <div class="card">

                <h2>تحسن المكافأة</h2>

                <div class="value">
                    ${number(improvement.reward_percent).toFixed(2)}%
                </div>

            </div>

            <div class="card">

                <h2>تحسن الدقة</h2>

                <div class="value">
                    ${number(improvement.accuracy_points).toFixed(2)}
                </div>

                <div class="small">
                    نقطة مئوية
                </div>

            </div>

        </div>


        <div class="card" style="margin-top:20px">

            <h2>🧠 المعرفة المكتسبة</h2>

            <div class="value">
                ${number(data.knowledge_size)}
            </div>

            <div class="small">
                عناصر معرفية
            </div>

        </div>


        <div class="card" style="margin-top:20px">

            <h2>💾 الذاكرة</h2>

            <div class="value">
                ${number(data.memory_size)}
            </div>

            <div class="small">
                تجربة محفوظة
            </div>

        </div>


        <div class="card" style="margin-top:20px">

            <h2>🧪 التدريب</h2>

            <p>
                عدد التجارب:
                <strong>${number(training.episodes)}</strong>
            </p>

            <p>
                متوسط مكافأة التدريب:
                <strong>
                ${number(training.average_reward).toFixed(4)}
                </strong>
            </p>

        </div>


        <div class="verdict">

            🔬 الحكم التجريبي

            <br><br>

            ${
                data.verdict
                ? "✅ " + data.verdict
                : "تم اكتمال الاختبار"
            }

        </div>

        `;

        status.innerHTML = "✅ اكتمل الاختبار";

    }

    catch (error) {

        console.error(error);

        status.innerHTML = `
        <div class="error">
        ❌ حدث خطأ أثناء تشغيل الاختبار
        <br><br>
        ${error.message}
        </div>
        `;

    }

    finally {

        button.disabled = false;

    }

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

    except Exception as e:

        return jsonify({
            "error": str(e)
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