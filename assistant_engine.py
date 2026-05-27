"""
assistant_engine.py — Heart Health Assistant response engine

Generates context-aware, educational responses using pattern matching.
No external LLM required — runs fully offline.
"""

import re
import random

DISCLAIMER = (
    "Disclaimer: This assistant provides educational information only "
    "and does not replace professional medical advice, diagnosis, or treatment."
)

# ── intent patterns ─────────────────────────────────────────────────────────

_INTENTS = {
    "emergency": [
        r"\bemergency\b",
        r"\bheart\s*attack\b",
        r"\b(chest\s*pain.*(severe|now|urgent|right\s*now|sudden))\b",
        r"\b(911|call\s*ambulance|emergency\s*room|er\b)",
    ],
    "greeting": [
        r"\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))\b",
    ],
    "how_platform_works": [
        r"\bhow\s.*(work|platform|app|tool|predictor)\b",
        r"\bwhat\s.*(this|platform|tool).*do\b",
        r"\bexplain\s.*(platform|tool|predictor)\b",
    ],
    "prediction_meaning": [
        r"\bhow\s.*(predict|interpret|read).*(result|risk)\b",
        r"\bexplain\s.*(prediction|results?|risk|probability|outcome)\b",
        r"\b(what\s.*mean|meaning\s.*result|interpret)\b",
        r"\bhow\s.*accurate\b",
        r"\breliable\b",
    ],
    "risk_levels": [
        r"\bwhat\s.*(risk\s*(level|category)|(high|medium|low)\s*risk)\b",
        r"\b(high|medium|low)\s*risk.*(mean|indicate)\b",
    ],
    "prediction_meaning": [
        r"\bhow\s.*(predict|interpret|read).*(result|risk)\b",
        r"\bexplain\s.*(prediction|results?|risk|probability|outcome)\b",
        r"\b(what\s.*(mean|result|score|probability)\s)|.*mean.*(predict|result)\b",
        r"\bhow\s.*accurate\b",
        r"\breliable\b",
        r"\b(understand|help\s.*understand).*(results?|prediction|risk)\b",
    ],
    "risk_factors": [
        r"\brisk\s*factor",
        r"\bwhat\s.*(cause|lead|increase).*heart\b",
        r"\bwhat\s.*(contribute|affect).*risk\b",
    ],
    "cholesterol": [
        r"\bcholesterol\b",
        r"\bchol\b",
        r"\b(ldl|hdl|lipid)\b",
    ],
    "blood_pressure": [
        r"\b(blood\s*pressure|hypertension|trestbps|bp)\b",
    ],
    "chest_pain": [
        r"\bchest\s*pain",
        r"\bcp\b",
        r"\bangina\b",
        r"\bexang\b",
    ],
    "heart_rate": [
        r"\b(heart\s*rate|max\s*heart|thalch|pulse)\b",
    ],
    "blood_sugar": [
        r"\b(blood\s*sugar|fasting\s*blood|fbs|diabetes|glucose)\b",
    ],
    "ecg": [
        r"\b(ecg|ekg|electrocardiogram|restecg|resting\s*ecg)\b",
    ],
    "oldpeak": [
        r"\b(oldpeak|st\s*depression|st\s*segment)\b",
    ],
    "thalassemia": [
        r"\b(thalassemia|thal)\b",
    ],
    "lifestyle": [
        r"\b(lifestyle|diet|exercise|nutrition|habit)\b",
        r"\bhow\s.*(prevent|reduce|improve|lower|decrease)\b",
        r"\b(food|foods|eat|eating|nutrition|workout|activity|exercise)\b",
        r"\b(good|healthy|better|healthier)\s.*(heart|health)\b",
        r"\b(improve|boost|strengthen).*heart\b",
    ],
    "smoking": [
        r"\bsmok(ing|e)\b",
    ],
    "stress": [
        r"\bstress\b",
        r"\banxiety\b",
        r"\bmindfulness\b",
    ],
    "sleep": [
        r"\bsleep\b",
    ],
    "doctor_advice": [
        r"\bshould\s*(i|we)\s*(see|consult|visit)\s*(a\s*)?doctor\b",
        r"\bwhen\s.*(see|consult|visit)\s*doctor\b",
        r"\bmedical\s*advice\b",
    ],
    "shap": [
        r"\b(shap|feature\s*importance|why\s.*(factor|feature).*(matter|affect|influence))\b",
        r"\bwhat\s.*(influenced|contributed|affected).*result\b",
    ],
    "dashboard": [
        r"\b(dashboard|chart|trend|analytics|statistics)\b",
        r"\bhow\s.*(my\s*)?(progress|history|change)\b",
    ],
    "doctor_tools": [
        r"\b(doctor|patient\s*list|note|assessment\s*overview)\b",
        r"\bhow\s.*(manage|review).*patient\b",
    ],
    "why_result": [
        r"\bwhy\s.*(did|do|is|was).*(risk|result|high|prediction)\b",
        r"\bwhy\s.*(got|get|received).*(high|risk|result)\b",
    ],
    "how_assessment_works": [
        r"\bhow\s.*(assessment|form|questionnaire|input|fill)\b",
        r"\bwhat\s.*(enter|provide|fill)\b",
        r"\bhelp\s.*(assessment|form|input|field)\b",
    ],
    "thanks": [
        r"\b(thank|thanks|appreciate)\b",
    ],
    "about_assistant": [
        r"\bwho\s.*(you|are)\b",
        r"\bwhat\s.*(are\s*you|your\s*purpose|can\s*you\s*do)\b",
        r"\btell\s*me\s*about\s*yourself\b",
    ],
    "age_risk": [
        r"\bage\b",
    ],
    "slope_st": [
        r"\bslope\b",
    ],
    "ca_vessels": [
        r"\b(ca|major\s*vessels|vessels|fluoroscopy)\b",
    ],
}


def _match_intent(message: str) -> str | None:
    msg_lower = message.lower()
    for intent, patterns in _INTENTS.items():
        for pat in patterns:
            if re.search(pat, msg_lower):
                return intent
    return None


# ── response templates ──────────────────────────────────────────────────────

_GREETING_RESPONSES = [
    "Hey there! I'm your Heart Health Assistant 💙 I can help you understand your results, explain any medical terms, or share tips for a healthier heart. What's on your mind?",
    "Hi, welcome! I'm here to make heart health easy to understand. Ask me about your prediction results, risk factors, or anything about keeping your heart happy and healthy.",
    "Hello! So glad you're here. Whether you're curious about your assessment results, want to know what a specific number means, or need some heart-friendly lifestyle advice — I'm ready to help. What would you like to explore?",
]

_HOW_PLATFORM_RESPONSES = [
    "Great question! This platform uses a machine learning model trained on real clinical data from the UCI Heart Disease dataset. You enter 13 health indicators — things like your age, blood pressure, cholesterol, and ECG results — and the model estimates your probability of heart disease. After the prediction, you'll also see a SHAP explanation that shows exactly which factors influenced your result the most. It's like having a smart assistant that helps you understand where your heart health stands.",
    "Think of it this way: you provide your health numbers, and the model compares them against thousands of real patient records to give you a personalized risk estimate. It uses a Gradient Boosting algorithm, which is quite clever at finding patterns in medical data. The result comes with a clear risk level (Low, Medium, or High) plus a breakdown of why the model gave that result — so nothing is hidden in a black box.",
]

_PREDICTION_MEANING_RESPONSES = [
    "Here's how to read your results: The number you see is your estimated probability of heart disease based on the indicators you entered. If your risk is Low (below 35%), your health numbers are generally in a favorable range. Medium risk (35–65%) means some areas could use a closer look — maybe a chat with your doctor. High risk (above 65%) means the pattern of your results aligns more closely with heart disease profiles, and we'd strongly encourage you to share these results with a healthcare professional.",
    "Your result is basically the model's best educated guess based on patterns learned from real medical data. It looks at all 13 of your inputs together and compares them to what it has seen before. The three risk levels help summarize where you fall: Low, Medium, or High. But the really useful part is the SHAP chart below your result — that's what shows you specifically which factors pushed your risk up or down. That's the part to pay attention to.",
    "A quick note about accuracy: this model was trained on clinical data and performs well for an educational screening tool. But no model is 100% accurate for everyone. Think of it as a helpful starting point for a conversation with your doctor — not a final diagnosis. Your doctor can look at your full health picture and give you proper medical guidance.",
]

_RISK_LEVELS_RESPONSES = [
    "The three risk levels help summarize where your results fall:\n\n• Low Risk (below 35%) — Your health indicators are generally in a favorable range. Keep up the good habits!\n• Medium Risk (35–65%) — Some factors could use attention. It's a good time to review your lifestyle and talk to your doctor.\n• High Risk (above 65%) — Your results align more closely with heart disease patterns. Please share these results with a healthcare professional.\n\nRemember, these are estimates based on patterns in clinical data — your doctor can give you a complete evaluation.",
    "Think of the risk level as a general indicator, not a final verdict. Low means your numbers look good overall, Medium means there are some things worth paying attention to, and High means your results follow a pattern that deserves medical follow-up. The most useful information is in the SHAP breakdown — it tells you exactly which factors are driving your result.",
]

_RISK_FACTORS_RESPONSES = [
    "There are several things that can affect your heart disease risk, and knowing them is the first step toward taking control:\n\n• Age — risk naturally goes up as we get older\n• Blood pressure — high readings put extra strain on your heart\n• Cholesterol — too much LDL can clog arteries over time\n• Smoking — it damages blood vessels in many ways\n• Diabetes — high blood sugar affects circulation\n• Chest pain type — certain kinds are stronger indicators\n• Exercise habits — staying active makes a real difference\n• Stress — chronic stress takes a toll on your heart\n\nThe good news? Many of these are things you can influence. Even small changes can add up.",
    "Heart disease risk isn't about just one thing — it's the combination of many factors that tells the full story. Some you can't change (like age and genetics), but many you can. That's why this platform looks at 13 different indicators together. It gives you a more complete picture than focusing on any single number.",
]

_CHOLESTEROL_RESPONSES = [
    "Cholesterol often gets a bad rap, but your body actually needs it! The key is the balance. LDL cholesterol (sometimes called 'bad' cholesterol) can build up in your artery walls and form plaque, which narrows the passageways for blood flow. HDL ('good' cholesterol) helps sweep away excess LDL. Your total cholesterol level is measured in mg/dL, and generally under 200 is considered desirable — but everyone's different. A heart-healthy diet (think more plants, less saturated fat), regular movement, and routine checkups can help keep your cholesterol in a good range.",
    "The 'chol' value in your assessment is your total serum cholesterol. It's just one piece of the puzzle — your doctor will also look at your LDL, HDL, and triglycerides to get the full picture. If you're concerned about your numbers, a simple blood test and a chat with your doctor can tell you a lot.",
]

_BLOOD_PRESSURE_RESPONSES = [
    "Blood pressure is the force of your blood pushing against your artery walls. It's shown as two numbers: systolic (the pressure when your heart beats) over diastolic (the pressure when your heart rests between beats). A normal reading is typically around 120/80 mmHg. When your blood pressure stays high over time (hypertension), it forces your heart to work harder and can damage your arteries. The good news is that lifestyle changes — cutting back on sodium, staying active, managing stress, and keeping a healthy weight — can make a real difference.",
    "The 'resting blood pressure' field in your assessment (trestbps) is your blood pressure measured when you're calm and at rest. If it's elevated, don't panic — one high reading doesn't mean hypertension. But it's worth monitoring regularly and discussing with your doctor, especially if it stays high.",
]

_CHEST_PAIN_RESPONSES = [
    "Chest pain is an important signal, and the assessment asks about it in two ways. 'Chest pain type' (cp) categorizes what kind of discomfort you experience — typical angina (related to your heart), atypical angina, non-anginal pain, or no pain at all. 'Exercise-induced angina' (exang) is chest pain that happens specifically during physical activity when your heart needs more oxygen. Both help the model understand your heart's response to stress. If you do experience any chest pain, please take it seriously and talk to a doctor.",
    "Angina is a specific type of chest discomfort caused by reduced blood flow to your heart muscle. It can feel like pressure, squeezing, or heaviness. Exercise-induced angina (exang) is particularly meaningful because it shows how your heart handles increased demand. This is an important thing to discuss with your healthcare provider if it applies to you.",
]

_HEART_RATE_RESPONSES = [
    "Your maximum heart rate (thalch) is the highest your heart rate gets during exercise or a stress test. It's a pretty good indicator of your cardiovascular fitness. Generally, a higher max heart rate during exercise is a positive sign — it means your heart can rise to the challenge. A lower-than-expected max heart rate during exertion could sometimes suggest reduced cardiac function. But keep in mind, your max heart rate naturally decreases as you age, and 'normal' varies from person to person.",
    "Think of your max heart rate as a measure of how well your heart responds when you ask it to work harder. If you're concerned about your numbers, a stress test monitored by a doctor can give you much more detailed information about your heart's performance.",
]

_BLOOD_SUGAR_RESPONSES = [
    "Fasting blood sugar (fbs) measures the amount of glucose in your blood after you haven't eaten for at least 8 hours. A level above 120 mg/dL (or 7 mmol/L) is considered elevated and could be a sign of diabetes or prediabetes. Over time, high blood sugar can damage your blood vessels and nerves, which increases your risk for heart disease. Managing your blood sugar through a balanced diet, regular exercise, and any medications your doctor prescribes is one of the most important things you can do for your heart.",
    "There's a strong link between diabetes and heart disease — that's why fasting blood sugar is one of the indicators in your assessment. If your number is high, it's definitely worth discussing with your doctor. The good news is that lifestyle changes can have a big impact on blood sugar control.",
]

_ECG_RESPONSES = [
    "An ECG (or EKG) records the electrical activity of your heart. It's a simple, painless test that can reveal a lot about your heart's health. The 'resting ECG' field in your assessment (restecg) captures what your heart's electrical pattern looks like while you're at rest. Results can be normal, show ST-T wave abnormalities, or indicate left ventricular hypertrophy (thickening of the heart muscle). Your doctor is the best person to explain what any ECG findings mean in your specific situation.",
    "Think of an ECG as a snapshot of your heart's electrical system. It can pick up irregular rhythms, past heart attacks, or signs that your heart is working under extra strain. It's a standard, routine test that gives your doctor valuable clues about your heart health.",
]

_OLDPEAK_RESPONSES = [
    "Oldpeak (ST depression) is a measurement taken from an ECG during a stress test — it shows how much the ST segment of your heart's electrical signal drops below the baseline when you're exercising. Higher values of ST depression can indicate that parts of your heart aren't getting enough blood flow during physical activity. It's considered one of the stronger predictors in heart disease assessment, which is why it's included in your evaluation.",
    "Think of it this way: when you exercise, your heart should handle the increased demand smoothly. ST depression on the ECG suggests there might be some reduced blood flow during exertion. If this is flagged in your results, it's definitely something to talk to your doctor about.",
]

_THALASSEMIA_RESPONSES = [
    "Thalassemia (thal) refers to a type of blood disorder that affects how your body produces hemoglobin — the protein that carries oxygen in your red blood cells. In your assessment, it's classified as normal, or as having a fixed or reversible defect. A 'reversible defect' is especially important because it means there are areas of your heart that get enough blood at rest but not during stress — which can be a sign of coronary artery disease. Your doctor can explain what this means for you personally.",
    "Thalassemia results are categorized in a specific way for this assessment. If your result shows a defect, your doctor may recommend follow-up testing to get a clearer picture of your coronary health.",
]

_LIFESTYLE_RESPONSES = [
    "The great news is that even small changes to your daily habits can have a meaningful impact on your heart health. Here are some of the most effective ones:\n\n• Eat a colorful diet — fruits, vegetables, whole grains, and lean proteins\n• Cut back on saturated fats, salty foods, and added sugars\n• Aim for about 30 minutes of moderate activity most days (a brisk walk counts!)\n• Try to keep a healthy weight\n• Prioritize sleep — 7 to 9 hours makes a real difference\n• Find ways to manage stress — even a few minutes of deep breathing helps\n• If you smoke, quitting is the single biggest gift you can give your heart\n\nThe key is consistency, not perfection. Small steps taken regularly add up.",
    "You don't need to overhaul your entire life at once. Start with one or two changes that feel manageable. Maybe it's swapping sugary drinks for water, taking a 15-minute walk after dinner, or adding an extra serving of vegetables to your plate. These might seem small, but over weeks and months, they genuinely lower your risk. And your doctor can help you build a plan that fits your specific situation.",
]

_SMOKING_RESPONSES = [
    "If you smoke, quitting is quite literally the best thing you can do for your heart. Smoking damages your blood vessels, reduces oxygen in your blood, and makes your heart work harder. Here's something encouraging: within just one year of quitting, your risk of heart disease drops by about 50%. Your body starts repairing itself almost immediately. If you're thinking about quitting, talk to your doctor about strategies that could help — whether it's nicotine replacement, medications, or support programs. You don't have to do it alone.",
    "Smoking affects nearly every part of your cardiovascular system. It's not just about your lungs — it's one of the biggest risk factors for heart disease, stroke, and high blood pressure. The earlier you quit, the more your heart can recover. Every cigarette not smoked is a win for your health.",
]

_STRESS_RESPONSES = [
    "Stress is something we all deal with, but when it becomes chronic, it can take a real toll on your heart. It can raise your blood pressure, increase inflammation in your body, and sometimes lead to unhealthy coping habits like poor eating or smoking. The good news is that managing stress doesn't have to be complicated. Even simple things — a short walk, deep breathing for a minute, calling a friend, or doing something you enjoy — can help lower your stress levels. Your heart will thank you.",
    "Think of stress management as part of your heart health routine, just like eating well or exercising. Even five minutes of mindfulness or stepping away from your screen can make a difference. Find what works for you and make it a regular habit.",
]

_SLEEP_RESPONSES = [
    "Sleep is when your body repairs itself, and that includes your heart and blood vessels. Consistently getting less than 7 hours or poor-quality sleep has been linked to higher risks of high blood pressure, heart disease, and even stroke. Aim for 7–9 hours of restful sleep each night. Simple habits like keeping a consistent bedtime, making your bedroom cool and dark, and putting away screens an hour before bed can improve your sleep quality significantly.",
    "If you're not sleeping well, your heart notices. Poor sleep can affect your blood pressure, inflammation levels, and even your appetite hormones. Prioritizing good sleep is one of the most underrated ways to protect your heart.",
]

_DOCTOR_ADVICE_RESPONSES = [
    "Absolutely — if you're concerned about your heart health, talking to a doctor is always the right call. This tool is designed for educational awareness, not to replace professional medical advice. Your doctor can look at your results alongside your personal medical history, run any necessary tests, and guide you on the best next steps. Think of your assessment as a conversation starter for that appointment.",
    "I'm here to help you understand your numbers and what they might mean, but I'm not a substitute for a real healthcare professional. If you received a High risk result, or if anything in your assessment worries you, please share it with your doctor. They can give you the personalized care and guidance you deserve.",
]

_SHAP_RESPONSES = [
    "SHAP stands for SHapley Additive exPlanations, but you can just think of it as 'the model explaining its reasoning.' After your prediction, you'll see a chart with bars for each health factor. Bars pointing to the right mean that factor increased your risk, while bars to the left decreased it. The longer the bar, the stronger that factor's influence. So if you see a big bar for 'cholesterol' pointing right, that tells you cholesterol was a major reason your risk went up. It's the model being transparent about why it gave you that result.",
    "Think of SHAP values as a breakdown of your prediction. The model looks at all your health numbers together, and SHAP shows you how much each one pushed the final result in one direction or another. This is super useful because it tells you which specific areas to focus on. If age and blood pressure are the biggest contributors, those are the things to discuss with your doctor.",
]

_DASHBOARD_RESPONSES = [
    "Your dashboard is like a personal heart health timeline. You can see all your past assessments, how your risk level has changed over time, and which factors have been most consistent in your results. This is really helpful for spotting trends — maybe your risk has been trending down since you started exercising more, or you can see how changes in your numbers affect your predictions over time.",
    "The charts in your dashboard give you a visual overview of your heart health journey. You can track your probability scores across different dates, see your risk distribution, and get a sense of your progress. It's a great tool to share with your doctor during checkups too.",
]

_DOCTOR_TOOLS_RESPONSES = [
    "The doctor dashboard gives you a comprehensive view of all patient assessments. You can browse individual patient profiles, review their risk history, and add clinical notes to specific predictions. The analytics section provides aggregated insights — you can see risk distribution across your patients, monthly trends, and even correlations between age and probability. It's designed to help you identify which patients might need earlier follow-up and track population-level patterns.",
    "As a doctor, you can use this platform to keep an eye on your patients' cardiovascular risk between visits. The patient profiles show detailed assessment history, and you can add professional notes to any prediction. The analytics page is particularly useful for spotting trends and identifying high-risk individuals who might benefit from proactive care.",
]

_HOW_ASSESSMENT_RESPONSES = [
    "The assessment asks for 13 health indicators that the model uses to estimate your heart disease risk. These include your age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar, resting ECG results, max heart rate during exercise, whether you have exercise-induced angina, ST depression (oldpeak), ST slope, number of major vessels visible on fluoroscopy, and thalassemia type. Each field has a small info icon you can click for more details. Just fill in the values you know, and for the most accurate results, use numbers from your latest health checkup if you have them.",
    "Don't worry if you're not sure about every single value — just fill in what you can. It's better to leave something blank than to guess. If you have your recent lab results or a health screening report handy, that makes it easier. The whole thing only takes a few minutes!",
]

_WHY_RESULT_RESPONSES = [
    "Your risk result is calculated by comparing your health numbers against patterns the model learned from thousands of real patient cases. If your result came back High, it means your combination of indicators is more similar to people who had heart disease in the training data. The most important thing to look at is the SHAP explanation below your result — it will tell you exactly which factors pushed your risk up the most. Common contributors to higher risk include elevated cholesterol, higher age, certain chest pain types, and abnormal ECG readings. Focus on the factors you can influence and discuss the rest with your doctor.",
    "The model looks at all 13 of your inputs together and weighs them based on their importance. If you got a higher risk result, some specific factors likely stood out. The SHAP chart breaks this down visually — you'll see exactly which measurements raised your risk and by how much. This is really valuable because it tells you and your doctor where to focus attention.",
]

_ACCURACY_RESPONSES = [
    "The model was trained on the UCI Heart Disease dataset using a Gradient Boosting classifier, which is a well-established machine learning algorithm. It performs reliably for educational screening purposes, but like any model, it's not perfect. Think of it as a helpful indicator that highlights areas worth discussing with your doctor — not as a medical diagnosis. Your doctor can combine these results with a physical exam, your medical history, and additional tests for a complete picture.",
    "While the model has been validated on clinical data and performs well, individual results can vary. The tool is designed for educational awareness and early detection support, not to replace clinical judgment. Always share your results with a healthcare professional who can interpret them in the context of your overall health.",
]

_EMERGENCY_RESPONSES = [
    "⚠️ Please stop — if you're having severe chest pain, trouble breathing, or any symptoms of a heart attack, call emergency services right now. Don't wait, don't use this tool, don't drive yourself. Call your local emergency number immediately. This is not something to handle alone.",
    "This assistant is not equipped for emergencies. If you think you're having a medical emergency — chest pain, difficulty breathing, or severe discomfort — please call your local emergency number immediately. Every second counts.",
]

_THANKS_RESPONSES = [
    "You're welcome! I'm really glad I could help 😊 Remember, every little step you take toward heart health matters. Feel free to come back anytime you have questions.",
    "Happy to help! Take good care of your heart — it works hard for you every single day 💙",
]

_ABOUT_ASSISTANT_RESPONSES = [
    "I'm the Heart Health Assistant, your AI-powered guide built right into this platform. My job is to help make heart health information clear and approachable. I can explain your prediction results, break down medical terms, walk you through SHAP values, and share practical lifestyle tips. One important thing to know: I'm not a doctor and I can't diagnose anything. But I'm here to help you understand your health better so you can have more informed conversations with your healthcare provider.",
]

_AGE_RESPONSES = [
    "Age is one of those risk factors we can't change — but knowing about it helps us stay on top of the things we can control. As we get older, our arteries naturally become less flexible, and the heart may not work quite as efficiently as it used to. That's why it becomes even more important to keep an eye on blood pressure, cholesterol, and lifestyle habits as the years go by. Being aware gives you the power to take action where it matters most.",
]

_SLOPE_RESPONSES = [
    "The ST slope describes the angle of your heart's electrical signal during the stress test portion of an ECG. It can be upsloping (which is generally considered normal), flat, or downsloping. A downsloping pattern during exercise can be a sign that some parts of your heart aren't getting enough blood flow when they're working harder. It's one of several indicators the model looks at to assess risk.",
    "Think of the ST segment as a line on your ECG. Normally it goes upward (upsloping) during exercise. If it goes flat or downward instead, that can be meaningful — it's something your doctor would want to know about.",
]

_CA_VESSELS_RESPONSES = [
    "The 'ca' field refers to the number of major coronary blood vessels (0 to 3) that show significant narrowing when examined with a special dye and X-ray (fluoroscopy). More affected vessels generally suggest more extensive coronary artery disease. This is determined through specialized imaging, not a standard blood test, so it may not be available for everyone. If this number is higher in your results, it's an important finding to discuss with your cardiologist.",
    "This measurement comes from an imaging procedure that looks at your coronary arteries directly. The number of vessels with significant blockages is one of the strongest predictors in heart disease assessment.",
]

_FALLBACK_RESPONSES = [
    "I want to make sure I help you well! Here are some things I can answer questions about:\n\n• How your prediction results work\n• Heart disease risk factors (cholesterol, blood pressure, etc.)\n• What specific medical terms in the assessment mean\n• Tips for a heart-healthy lifestyle\n• How SHAP explanations work\n• Navigating your dashboard or history\n\nCould you try rephrasing your question? I'm here to help!",
    "I'm not sure I caught that — sorry! Let me tell you what I can help with:\n\n• Understanding your risk result\n• Explaining medical terms from your assessment\n• Heart-friendly diet and exercise tips\n• How the prediction model works\n• What the charts and trends in your dashboard mean\n\nGo ahead and ask in a different way — I'll do my best to help!",
]


def _build_context_header(context: dict) -> str:
    page = context.get("page", "unknown")
    risk_level = context.get("risk_level")
    probability = context.get("probability")

    header_parts = ["You are on the"]
    if page == "index" or page == "home":
        header_parts.append("homepage.")
    elif page == "predict" or page == "assessment":
        header_parts.append("assessment page.")
    elif page == "result" or page == "results":
        header_parts.append("results page.")
        if risk_level:
            header_parts.append(f"The user's risk level is {risk_level}")
        if probability is not None:
            header_parts.append(f"with a probability of {probability}%")
    elif page == "history":
        header_parts.append("history page.")
    elif page == "dashboard" or page == "user_dashboard":
        header_parts.append("user dashboard.")
    elif page == "doctor_dashboard":
        header_parts.append("doctor dashboard.")
    elif page == "analytics":
        header_parts.append("analytics page.")
    elif page == "about":
        header_parts.append("about page.")
    else:
        header_parts.append("the platform.")

    return " ".join(header_parts)


def generate_response(message: str, context: dict | None = None) -> str:
    context = context or {}
    context_header = _build_context_header(context)

    intent = _match_intent(message)

    if intent == "emergency":
        return random.choice(_EMERGENCY_RESPONSES)

    if intent == "greeting":
        return random.choice(_GREETING_RESPONSES)

    if intent == "how_platform_works":
        return f"{random.choice(_HOW_PLATFORM_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "prediction_meaning":
        return f"{random.choice(_PREDICTION_MEANING_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "risk_levels":
        return f"{random.choice(_RISK_LEVELS_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "risk_factors":
        return f"{random.choice(_RISK_FACTORS_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "cholesterol":
        return f"{random.choice(_CHOLESTEROL_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "blood_pressure":
        return f"{random.choice(_BLOOD_PRESSURE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "chest_pain":
        return f"{random.choice(_CHEST_PAIN_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "heart_rate":
        return f"{random.choice(_HEART_RATE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "blood_sugar":
        return f"{random.choice(_BLOOD_SUGAR_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "ecg":
        return f"{random.choice(_ECG_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "oldpeak":
        return f"{random.choice(_OLDPEAK_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "thalassemia":
        return f"{random.choice(_THALASSEMIA_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "lifestyle":
        return f"{random.choice(_LIFESTYLE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "smoking":
        return f"{random.choice(_SMOKING_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "stress":
        return f"{random.choice(_STRESS_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "sleep":
        return f"{random.choice(_SLEEP_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "doctor_advice":
        return f"{random.choice(_DOCTOR_ADVICE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "shap":
        return f"{random.choice(_SHAP_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "dashboard":
        return f"{random.choice(_DASHBOARD_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "doctor_tools":
        return f"{random.choice(_DOCTOR_TOOLS_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "how_assessment_works":
        return f"{random.choice(_HOW_ASSESSMENT_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "why_result":
        return f"{random.choice(_WHY_RESULT_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "thanks":
        return random.choice(_THANKS_RESPONSES)

    if intent == "about_assistant":
        return f"{random.choice(_ABOUT_ASSISTANT_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "age_risk":
        return f"{random.choice(_AGE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "slope_st":
        return f"{random.choice(_SLOPE_RESPONSES)}\n\n{DISCLAIMER}"

    if intent == "ca_vessels":
        return f"{random.choice(_CA_VESSELS_RESPONSES)}\n\n{DISCLAIMER}"

    return f"{random.choice(_FALLBACK_RESPONSES)}\n\n{DISCLAIMER}"
