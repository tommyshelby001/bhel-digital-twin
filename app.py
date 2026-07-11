from flask import Flask, render_template, jsonify, redirect, url_for, request, session, flash, Response
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import joblib
import os
import io
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "bhel-upes-dt-2026-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "turbine_model.pkl")
MODEL = joblib.load(MODEL_PATH)

METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics_report.json")
try:
    with open(METRICS_PATH) as f:
        MODEL_METRICS = json.load(f)
except:
    MODEL_METRICS = {"accuracy": 0.977, "n_samples": 5000, "cv_mean": 0.981,
                     "f1_score": 0.9418, "roc_auc": 0.9547,
                     "feature_importance": {"steam_temp_C":0.1662,"steam_pressure_kgcm2":0.0484,
                     "vibration_mms":0.2335,"bearing_temp_C":0.4148,"rpm":0.0247,
                     "oil_pressure_kgcm2":0.0964,"load_percent":0.016}}

USERS = {
    "admin":     {"password": generate_password_hash("bhel@admin123"),  "role": "admin",    "name": "System Administrator", "dept": "IT / Control Systems",    "last_login": None},
    "operator1": {"password": generate_password_hash("operator@123"),   "role": "operator", "name": "Rajesh Kumar",          "dept": "Turbine Operations",       "last_login": None},
    "engineer1": {"password": generate_password_hash("engineer@123"),   "role": "engineer", "name": "Priya Sharma",          "dept": "Mechanical Engineering",    "last_login": None},
}

ALERT_LOG = []
MACHINES = {
    "TG-01": {"name": "Turbo Generator Unit 1", "capacity": "500 MW", "status": "online"},
    "TG-02": {"name": "Turbo Generator Unit 2", "capacity": "500 MW", "status": "online"},
    "TG-03": {"name": "Turbo Generator Unit 3", "capacity": "210 MW", "status": "maintenance"},
}
FEATURES = ["steam_temp_C","steam_pressure_kgcm2","vibration_mms","bearing_temp_C","rpm","oil_pressure_kgcm2","load_percent"]
MACHINE_STATE = {mid: {"tick": 0, "history": []} for mid in MACHINES}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access only.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

def get_sensor_reading(machine_id):
    state = MACHINE_STATE[machine_id]
    state["tick"] += 1
    cycle = state["tick"] % 60
    severity = 0 if cycle < 35 else min((cycle - 35) / 25, 1.0)
    r = {
        "steam_temp_C":         round(float(np.random.normal(540 + severity*35, 4)), 1),
        "steam_pressure_kgcm2": round(float(np.random.normal(170 - severity*15, 3)), 1),
        "vibration_mms":        round(max(float(np.random.normal(2.5 + severity*7, 0.5)), 0), 2),
        "bearing_temp_C":       round(float(np.random.normal(65 + severity*30, 3)), 1),
        "rpm":                  round(float(np.random.normal(3000 - severity*50, 10)), 0),
        "oil_pressure_kgcm2":   round(max(float(np.random.normal(2.2 - severity*0.8, 0.1)), 0), 2),
        "load_percent":         round(min(max(float(np.random.normal(85 - severity*20, 5)), 0), 100), 1),
    }
    X = pd.DataFrame([[r[f] for f in FEATURES]], columns=FEATURES)
    pred = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0]
    classes = list(MODEL.classes_)
    prob = round(float(proba[classes.index(1)]) * 100, 1) if 1 in classes else 0.0
    status = "FAILURE_RISK" if pred == 1 else "SAFE"
    if prob < 20:   rul, rul_label = 90, "Excellent"
    elif prob < 45: rul, rul_label = 45, "Monitor"
    elif prob < 70: rul, rul_label = 14, "Schedule Maintenance"
    else:           rul, rul_label = 3,  "URGENT"
    entry = {**r, "status": status, "failure_probability": prob,
             "rul_days": rul, "rul_label": rul_label,
             "timestamp": datetime.now().strftime("%H:%M:%S"),
             "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    state["history"].append(entry)
    if len(state["history"]) > 500:
        state["history"].pop(0)
    if status == "FAILURE_RISK" and (
        not ALERT_LOG or ALERT_LOG[-1]["machine"] != machine_id or
        ALERT_LOG[-1]["status"] != "FAILURE_RISK"
    ):
        ALERT_LOG.append({
            "id": len(ALERT_LOG) + 1,
            "machine": machine_id,
            "machine_name": MACHINES[machine_id]["name"],
            "message": f"High vibration ({r['vibration_mms']} mm/s) & bearing temp ({r['bearing_temp_C']}°C)",
            "prob": prob, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "acknowledged": False, "status": "FAILURE_RISK"
        })
    return entry

# ── ROUTES ──
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        user = USERS.get(u)
        if user and check_password_hash(user["password"], p):
            session.update({"user":u,"role":user["role"],"name":user["name"],"dept":user["dept"]})
            USERS[u]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",
        user=session["name"], role=session["role"], dept=session["dept"],
        machines=MACHINES, active_alerts=sum(1 for a in ALERT_LOG if not a["acknowledged"]),
        metrics=MODEL_METRICS)

@app.route("/history")
@login_required
def history():
    return render_template("history.html",
        user=session["name"], role=session["role"], machines=MACHINES,
        active_alerts=sum(1 for a in ALERT_LOG if not a["acknowledged"]))

@app.route("/alerts")
@login_required
def alerts():
    return render_template("alerts.html",
        alerts=list(reversed(ALERT_LOG)),
        user=session["name"], role=session["role"],
        active_alerts=sum(1 for a in ALERT_LOG if not a["acknowledged"]))

@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html",
        users=USERS, machines=MACHINES,
        user=session["name"], role=session["role"],
        active_alerts=sum(1 for a in ALERT_LOG if not a["acknowledged"]),
        metrics=MODEL_METRICS)

# ── API ──
@app.route("/api/live/<machine_id>")
@login_required
def api_live(machine_id):
    if machine_id not in MACHINES:
        return jsonify({"error": "Not found"}), 404
    if MACHINES[machine_id]["status"] == "maintenance":
        return jsonify({"status": "MAINTENANCE", "machine_id": machine_id})
    return jsonify(get_sensor_reading(machine_id))

@app.route("/api/history/<machine_id>")
@login_required
def api_history(machine_id):
    if machine_id not in MACHINES:
        return jsonify({"error": "Machine not found"}), 404
    limit = request.args.get("limit", request.args.get("n", 100), type=int)
    limit = max(1, min(limit, 500))
    history = MACHINE_STATE.get(machine_id, {}).get("history", [])
    data = history[-limit:]

    stats = {}
    if data:
        feature_keys = ["steam_temp_C", "steam_pressure_kgcm2", "vibration_mms",
                         "bearing_temp_C", "rpm", "oil_pressure_kgcm2", "load_percent"]
        for f in feature_keys:
            vals = [d[f] for d in data if f in d]
            if vals:
                stats[f] = {
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "avg": round(sum(vals) / len(vals), 2)
                }
    risk_count = sum(1 for d in data if d.get("status") == "FAILURE_RISK")

    return jsonify({
        "machine_id": machine_id,
        "count": len(data),
        "data": data,
        "stats": stats,
        "risk_readings": risk_count,
        "safe_readings": len(data) - risk_count
    })

@app.route("/api/history/<machine_id>/csv")
@login_required
def api_history_csv(machine_id):
    history = MACHINE_STATE.get(machine_id, {}).get("history", [])
    if not history:
        return "No data", 404
    df = pd.DataFrame(history)
    csv_data = df.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={machine_id}_history.csv"})

@app.route("/api/alerts/acknowledge/<int:alert_id>", methods=["POST"])
@login_required
def acknowledge_alert(alert_id):
    for a in ALERT_LOG:
        if a["id"] == alert_id:
            a.update({"acknowledged": True, "ack_by": session["name"],
                      "ack_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/summary")
@login_required
def api_summary():
    online = sum(1 for m in MACHINES.values() if m["status"]=="online")
    return jsonify({"total_machines": len(MACHINES), "online": online,
        "maintenance": len(MACHINES)-online,
        "active_alerts": sum(1 for a in ALERT_LOG if not a["acknowledged"]),
        "total_alerts": len(ALERT_LOG)})

@app.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(MODEL_METRICS)

# ── PDF REPORT GENERATOR ──
@app.route("/api/report/<machine_id>")
@login_required
def generate_report(machine_id):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        machine = MACHINES.get(machine_id, {"name": machine_id, "capacity": "N/A"})
        history = MACHINE_STATE.get(machine_id, {}).get("history", [])
        latest = history[-1] if history else {}

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            rightMargin=20*mm, leftMargin=20*mm,
            topMargin=20*mm, bottomMargin=20*mm)

        # Colors
        BHEL_DARK  = colors.HexColor("#0a1520")
        BHEL_AMBER = colors.HexColor("#f59e0b")
        BHEL_CYAN  = colors.HexColor("#06b6d4")
        BHEL_GREEN = colors.HexColor("#10b981")
        BHEL_RED   = colors.HexColor("#ef4444")
        GRAY       = colors.HexColor("#4a6a80")
        LIGHT_BG   = colors.HexColor("#f0f4f8")

        styles = getSampleStyleSheet()
        story = []

        def S(text, size=10, bold=False, color=colors.black, align=TA_LEFT, space_after=4):
            style = ParagraphStyle("custom", fontSize=size, fontName="Helvetica-Bold" if bold else "Helvetica",
                textColor=color, alignment=align, spaceAfter=space_after)
            return Paragraph(text, style)

        # ── HEADER ──
        header_data = [[
            S("BHEL", 28, bold=True, color=BHEL_AMBER),
            S("BHARAT HEAVY ELECTRICALS LIMITED<br/>HEEP Unit · Haridwar, Uttarakhand<br/>AI Digital Twin System v2.0", 9, color=GRAY, align=TA_RIGHT)
        ]]
        header_table = Table(header_data, colWidths=[80*mm, 90*mm])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BACKGROUND", (0,0), (-1,-1), BHEL_DARK),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING", (0,0), (0,0), 12),
            ("RIGHTPADDING", (-1,-1), (-1,-1), 12),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6*mm))

        # ── TITLE ──
        story.append(S("MACHINE HEALTH & PREDICTIVE MAINTENANCE REPORT", 16, bold=True, color=BHEL_DARK, align=TA_CENTER, space_after=2))
        story.append(S(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')} | Operator: {session.get('name','N/A')}", 9, color=GRAY, align=TA_CENTER, space_after=6))
        story.append(HRFlowable(width="100%", thickness=2, color=BHEL_AMBER))
        story.append(Spacer(1, 4*mm))

        # ── MACHINE INFO ──
        story.append(S("1. MACHINE IDENTIFICATION", 12, bold=True, color=BHEL_DARK, space_after=3))
        info_data = [
            ["Machine ID", machine_id, "Unit Name", machine["name"]],
            ["Capacity", machine["capacity"], "Plant", "HEEP Haridwar"],
            ["Status", machine.get("status","N/A").upper(), "Report Date", datetime.now().strftime("%d-%m-%Y")],
        ]
        info_table = Table(info_data, colWidths=[35*mm, 55*mm, 35*mm, 45*mm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
            ("BACKGROUND", (2,0), (2,-1), LIGHT_BG),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d8e0")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 5*mm))

        # ── AI STATUS ──
        story.append(S("2. AI PREDICTION STATUS", 12, bold=True, color=BHEL_DARK, space_after=3))
        ai_status = latest.get("status", "N/A")
        ai_prob = latest.get("failure_probability", 0)
        rul = latest.get("rul_days", 90)
        status_color = BHEL_RED if ai_status == "FAILURE_RISK" else BHEL_GREEN

        ai_data = [
            [S("AI STATUS", 9, bold=True), S(ai_status.replace("_"," "), 11, bold=True, color=status_color),
             S("FAILURE PROB", 9, bold=True), S(f"{ai_prob}%", 11, bold=True, color=BHEL_AMBER)],
            [S("RUL ESTIMATE", 9, bold=True), S(f"{rul} Days", 11, bold=True, color=BHEL_CYAN),
             S("CLASSIFIER", 9, bold=True), S("Random Forest (scikit-learn)", 9)],
        ]
        ai_table = Table(ai_data, colWidths=[40*mm, 50*mm, 40*mm, 40*mm])
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), BHEL_DARK),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#1c3045")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 5*mm))

        # ── SENSOR READINGS ──
        story.append(S("3. LIVE SENSOR READINGS", 12, bold=True, color=BHEL_DARK, space_after=3))
        SENSOR_LABELS = {
            "steam_temp_C": ("Steam Temperature","°C", 500, 560),
            "steam_pressure_kgcm2": ("Steam Pressure","kg/cm²", 145, 190),
            "vibration_mms": ("Vibration","mm/s", 0, 5.5),
            "bearing_temp_C": ("Bearing Temperature","°C", 50, 82),
            "rpm": ("Rotor Speed","RPM", 2920, 3050),
            "oil_pressure_kgcm2": ("Oil Pressure","kg/cm²", 1.4, 3),
            "load_percent": ("Load","%", 0, 100),
        }
        sensor_header = [S("Parameter",9,bold=True), S("Value",9,bold=True), S("Unit",9,bold=True), S("Normal Range",9,bold=True), S("Status",9,bold=True)]
        sensor_rows = [sensor_header]
        for k, (label, unit, lo, hi) in SENSOR_LABELS.items():
            val = latest.get(k, "--")
            try:
                val_f = float(val)
                ok = lo <= val_f <= hi
                st = S("NORMAL",8,color=BHEL_GREEN) if ok else S("WARNING",8,color=BHEL_RED,bold=True)
            except:
                st = S("N/A",8)
            sensor_rows.append([
                Paragraph(label, styles["Normal"]),
                S(str(val), 9, bold=True),
                S(unit, 9, color=GRAY),
                S(f"{lo} – {hi}", 9, color=GRAY),
                st
            ])
        sensor_table = Table(sensor_rows, colWidths=[55*mm, 25*mm, 20*mm, 35*mm, 25*mm])
        sensor_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BHEL_DARK),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d8e0")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(sensor_table)
        story.append(Spacer(1, 5*mm))

        # ── FEATURE IMPORTANCE ──
        story.append(S("4. AI MODEL — FEATURE IMPORTANCE", 12, bold=True, color=BHEL_DARK, space_after=3))
        story.append(S("Parameters ranked by contribution to failure prediction (Random Forest feature importance):", 9, color=GRAY, space_after=4))
        fi = MODEL_METRICS.get("feature_importance", {})
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fi_header = [S("Sensor Parameter",9,bold=True), S("Importance Score",9,bold=True), S("Contribution",9,bold=True)]
        fi_rows = [fi_header]
        fi_labels = {"steam_temp_C":"Steam Temperature","steam_pressure_kgcm2":"Steam Pressure",
                     "vibration_mms":"Vibration","bearing_temp_C":"Bearing Temperature",
                     "rpm":"Rotor Speed","oil_pressure_kgcm2":"Oil Pressure","load_percent":"Load"}
        for k, v in fi_sorted:
            bar = "█" * int(v * 30)
            fi_rows.append([
                S(fi_labels.get(k,k), 9),
                S(f"{v:.4f}", 9, bold=True, color=BHEL_AMBER),
                S(f"{bar} {v*100:.1f}%", 8, color=BHEL_CYAN)
            ])
        fi_table = Table(fi_rows, colWidths=[60*mm, 40*mm, 70*mm])
        fi_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BHEL_DARK),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d8e0")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(fi_table)
        story.append(Spacer(1, 5*mm))

        # ── MODEL PERFORMANCE ──
        story.append(S("5. ML MODEL PERFORMANCE METRICS", 12, bold=True, color=BHEL_DARK, space_after=3))
        m = MODEL_METRICS
        perf_data = [
            [S("Accuracy",9,bold=True), S(f"{m.get('accuracy',0)*100:.1f}%",11,bold=True,color=BHEL_GREEN),
             S("Precision",9,bold=True), S(f"{m.get('precision',0)*100:.1f}%",11,bold=True,color=BHEL_CYAN)],
            [S("Recall",9,bold=True), S(f"{m.get('recall',0)*100:.1f}%",11,bold=True,color=BHEL_CYAN),
             S("F1 Score",9,bold=True), S(f"{m.get('f1_score',0)*100:.1f}%",11,bold=True,color=BHEL_CYAN)],
            [S("ROC-AUC",9,bold=True), S(f"{m.get('roc_auc',0):.4f}",11,bold=True,color=BHEL_AMBER),
             S("CV Score (5-fold)",9,bold=True), S(f"{m.get('cv_mean',0)*100:.1f}%",11,bold=True,color=BHEL_AMBER)],
            [S("Training Samples",9,bold=True), S(f"{m.get('n_train',4000):,}",11,bold=True,color=colors.black),
             S("Test Samples",9,bold=True), S(f"{m.get('n_test',1000):,}",11,bold=True,color=colors.black)],
        ]
        perf_table = Table(perf_data, colWidths=[45*mm, 45*mm, 45*mm, 35*mm])
        perf_table.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, LIGHT_BG]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#d0d8e0")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 5*mm))

        # ── RECOMMENDATIONS ──
        story.append(S("6. MAINTENANCE RECOMMENDATIONS", 12, bold=True, color=BHEL_DARK, space_after=3))
        if ai_status == "FAILURE_RISK":
            recs = [
                "IMMEDIATE ACTION REQUIRED: Schedule emergency maintenance within 24-48 hours.",
                "Inspect bearing assembly for excessive wear or lubrication failure.",
                "Check vibration isolation mounts and shaft alignment.",
                "Review oil pressure and lubrication system integrity.",
                "Do not increase machine load until maintenance is completed.",
            ]
            rec_color = BHEL_RED
        elif rul <= 14:
            recs = [
                "Schedule preventive maintenance within the next 14 days.",
                "Perform bearing inspection and re-lubrication.",
                "Monitor vibration levels closely — trend shows upward movement.",
                "Check steam seal integrity and blade condition.",
            ]
            rec_color = BHEL_AMBER
        else:
            recs = [
                "Machine operating within normal parameters — no immediate action required.",
                "Continue scheduled preventive maintenance as per maintenance calendar.",
                "Next inspection recommended in 30 days or at next scheduled outage.",
                "Maintain monitoring for any parameter deviations.",
            ]
            rec_color = BHEL_GREEN
        for i, rec in enumerate(recs, 1):
            story.append(S(f"{i}. {rec}", 9, color=rec_color if i==1 and ai_status=="FAILURE_RISK" else colors.HexColor("#1a2a3a"), space_after=3))
        story.append(Spacer(1, 4*mm))

        # ── FOOTER ──
        story.append(HRFlowable(width="100%", thickness=1, color=BHEL_AMBER))
        story.append(Spacer(1, 3*mm))
        footer_data = [[
            S("BHEL Digital Twin System v2.0\nVocational Training Project — UPES × BHEL 2026", 8, color=GRAY),
            S(f"Report generated by: {session.get('name','N/A')}\nHEEP Haridwar | {datetime.now().strftime('%d %b %Y')}", 8, color=GRAY, align=TA_RIGHT)
        ]]
        footer_table = Table(footer_data, colWidths=[90*mm, 80*mm])
        footer_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(footer_table)

        doc.build(story)
        buf.seek(0)
        fname = f"BHEL_{machine_id}_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        return Response(buf.read(), mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={fname}"})

    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
