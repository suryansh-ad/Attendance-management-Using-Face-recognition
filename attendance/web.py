from __future__ import annotations

from datetime import date
import sqlite3

from flask import Flask, flash, redirect, render_template, request, url_for

from .config import BASE_DIR
from .database import (
    create_student,
    create_subject,
    delete_student,
    delete_subject,
    enroll_student_in_subject,
    enrolled_student_count,
    fetch_attendance_logs,
    fetch_all_students,
    fetch_all_subjects,
    fetch_student,
    fetch_student_by_code,
    fetch_student_by_identity,
    fetch_subject,
    fetch_subject_attendance_logs,
    fetch_subject_attendance_logs_by_status,
    fetch_subject_attendance_record,
    fetch_subject_by_code,
    fetch_subject_roster_statuses,
    init_db,
    initialize_subject_attendance,
    is_student_enrolled_in_subject,
    mark_subject_attendance_present,
    remove_student_from_subject,
    student_count,
    subject_attendance_summary,
    subject_count,
)
from .face_engine import FaceEncodingStore, enroll_from_camera, recognize_from_camera


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["SECRET_KEY"] = "attendance-secret-key"
    init_db()

    def _selected_subject_context():
        subjects = fetch_all_subjects()
        selected_subject_id_raw = request.args.get("subject_id", "").strip()
        session_date = request.args.get("session_date", "").strip() or date.today().isoformat()
        attendance_filter = request.args.get("attendance_filter", "").strip().title()
        if attendance_filter not in {"Present", "Absent"}:
            attendance_filter = "All"

        selected_subject = None
        if selected_subject_id_raw:
            try:
                selected_subject = fetch_subject(int(selected_subject_id_raw))
            except ValueError:
                selected_subject = None

        selected_subject_id = int(selected_subject["id"]) if selected_subject else None
        roster = fetch_subject_roster_statuses(selected_subject_id, session_date) if selected_subject_id else []
        attendance_logs = (
            fetch_subject_attendance_logs_by_status(
                selected_subject_id,
                session_date,
                None if attendance_filter == "All" else attendance_filter,
            )
            if selected_subject_id
            else []
        )
        summary = (
            subject_attendance_summary(selected_subject_id, session_date)
            if selected_subject_id
            else {"present": 0, "absent": 0}
        )
        enrolled_count = enrolled_student_count(selected_subject_id) if selected_subject_id else 0

        return {
            "subjects": subjects,
            "selected_subject": selected_subject,
            "selected_subject_id": selected_subject_id,
            "session_date": session_date,
            "attendance_filter": attendance_filter,
            "subject_roster": roster,
            "subject_attendance_logs": attendance_logs,
            "subject_summary": summary,
            "enrolled_in_selected_subject": enrolled_count,
        }

    @app.route("/")
    def dashboard():
        subject_context = _selected_subject_context()
        return render_template(
            "dashboard.html",
            students=fetch_all_students(),
            attendance_logs=fetch_attendance_logs(),
            total_students=student_count(),
            total_subjects=subject_count(),
            today=subject_context["session_date"],
            **subject_context,
        )

    @app.post("/students")
    def add_student():
        student_code = request.form.get("student_code", "").strip()
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()

        if not all([student_code, full_name, department]):
            flash("Please fill in student code, full name, and department.", "error")
            return redirect(url_for("dashboard"))

        existing_student = fetch_student_by_code(student_code)
        if existing_student is not None:
            flash(
                f"Student code {student_code} is already enrolled. Delete that profile first if you want to add again.",
                "error",
            )
            return redirect(url_for("dashboard"))

        existing_identity = fetch_student_by_identity(full_name, department)
        if existing_identity is not None:
            flash(
                f"{full_name} from {department} is already added as student code {existing_identity['student_code']}.",
                "error",
            )
            return redirect(url_for("dashboard"))

        try:
            student_id = create_student(student_code, full_name, department)
        except Exception as exc:
            flash(f"Could not add student: {exc}", "error")
            return redirect(url_for("dashboard"))

        try:
            encoding = enroll_from_camera()
            matched_student_id = encoding.duplicate_student_id or FaceEncodingStore().find_matching_student(encoding)
            if matched_student_id is not None:
                matched_student = fetch_student(matched_student_id)
                duplicate_label = (
                    f"{matched_student['full_name']} ({matched_student['student_code']})"
                    if matched_student is not None
                    else "an existing student"
                )
                FaceEncodingStore().delete_student_encoding(student_id)
                delete_student(student_id)
                flash(f"This face is already enrolled as {duplicate_label}.", "error")
                return redirect(url_for("dashboard"))
            FaceEncodingStore().save_student_encoding(student_id, encoding)
            flash(f"Student {full_name} enrolled successfully.", "success")
        except Exception as exc:
            FaceEncodingStore().delete_student_encoding(student_id)
            delete_student(student_id)
            flash(f"Student created, but face enrollment failed: {exc}", "error")

        return redirect(url_for("dashboard"))

    @app.post("/subjects")
    def add_subject():
        subject_code = request.form.get("subject_code", "").strip()
        subject_name = request.form.get("subject_name", "").strip()

        if not subject_code or not subject_name:
            flash("Please fill in subject code and subject name.", "error")
            return redirect(url_for("dashboard"))

        if fetch_subject_by_code(subject_code) is not None:
            flash(f"Subject code {subject_code} already exists.", "error")
            return redirect(url_for("dashboard"))

        try:
            create_subject(subject_code, subject_name)
            flash(f"Subject {subject_name} created successfully.", "success")
        except sqlite3.IntegrityError as exc:
            flash(f"Could not add subject: {exc}", "error")

        return redirect(url_for("dashboard"))

    @app.post("/subjects/<int:subject_id>/delete")
    def remove_subject(subject_id: int):
        subject = fetch_subject(subject_id)
        if subject is None:
            flash("Subject not found.", "error")
            return redirect(url_for("dashboard"))

        delete_subject(subject_id)
        flash(f"Deleted subject {subject['subject_name']}.", "success")
        return redirect(url_for("dashboard"))

    @app.post("/subjects/enrollments")
    def add_subject_enrollment():
        student_id_raw = request.form.get("student_id", "").strip()
        subject_id_raw = request.form.get("subject_id", "").strip()

        if not student_id_raw or not subject_id_raw:
            flash("Choose both a student and a subject.", "error")
            return redirect(url_for("dashboard"))

        try:
            student_id = int(student_id_raw)
            subject_id = int(subject_id_raw)
        except ValueError:
            flash("Invalid student or subject selection.", "error")
            return redirect(url_for("dashboard"))

        student = fetch_student(student_id)
        subject = fetch_subject(subject_id)
        if student is None or subject is None:
            flash("Student or subject not found.", "error")
            return redirect(url_for("dashboard"))

        if not enroll_student_in_subject(student_id, subject_id):
            flash(f"{student['full_name']} is already enrolled in {subject['subject_name']}.", "error")
            return redirect(url_for("dashboard", subject_id=subject_id))

        flash(f"Enrolled {student['full_name']} in {subject['subject_name']}.", "success")
        return redirect(url_for("dashboard", subject_id=subject_id))

    @app.post("/subjects/<int:subject_id>/students/<int:student_id>/remove")
    def remove_subject_enrollment(student_id: int, subject_id: int):
        subject = fetch_subject(subject_id)
        student = fetch_student(student_id)
        if subject is None or student is None:
            flash("Subject or student not found.", "error")
            return redirect(url_for("dashboard"))

        if remove_student_from_subject(student_id, subject_id):
            flash(f"Removed {student['full_name']} from {subject['subject_name']}.", "success")
        else:
            flash(f"{student['full_name']} was not enrolled in {subject['subject_name']}.", "error")
        return redirect(url_for("dashboard", subject_id=subject_id))

    @app.post("/subjects/<int:subject_id>/sessions")
    def start_subject_session(subject_id: int):
        session_date = request.form.get("session_date", "").strip() or date.today().isoformat()
        subject = fetch_subject(subject_id)
        if subject is None:
            flash("Subject not found.", "error")
            return redirect(url_for("dashboard"))

        enrolled_count = initialize_subject_attendance(subject_id, session_date)
        if enrolled_count == 0:
            flash(f"No students are enrolled in {subject['subject_name']} yet.", "error")
        else:
            flash(
                f"Session ready for {subject['subject_name']} on {session_date}. "
                f"{enrolled_count} enrolled students are currently marked absent until scanned present.",
                "success",
            )
        return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

    @app.post("/students/<int:student_id>/delete")
    def remove_student(student_id: int):
        student = fetch_student(student_id)
        if student is None:
            flash("Student not found.", "error")
            return redirect(url_for("dashboard"))

        FaceEncodingStore().delete_student_encoding(student_id)
        delete_student(student_id)
        flash(f"Deleted profile for {student['full_name']}.", "success")
        return redirect(url_for("dashboard"))

    @app.post("/subjects/<int:subject_id>/attendance/scan")
    def scan_attendance(subject_id: int):
        session_date = request.form.get("session_date", "").strip() or date.today().isoformat()
        subject = fetch_subject(subject_id)
        if subject is None:
            flash("Subject not found.", "error")
            return redirect(url_for("dashboard"))

        enrolled_count = initialize_subject_attendance(subject_id, session_date)
        if enrolled_count == 0:
            flash(f"No students are enrolled in {subject['subject_name']} yet.", "error")
            return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

        try:
            result = recognize_from_camera()
            if result is None or not result.recognized:
                flash((result.message if result else "Student is not recognized."), "error")
                return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

            student = fetch_student(result.student_id)
            if student is None:
                flash("Matched face was not linked to a student record.", "error")
                return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

            if not is_student_enrolled_in_subject(result.student_id, subject_id):
                flash(
                    f"{student['full_name']} is not enrolled in {subject['subject_name']}. "
                    "Attendance was not marked.",
                    "error",
                )
                return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

            existing_record = fetch_subject_attendance_record(result.student_id, subject_id, session_date)
            if existing_record is not None and existing_record["status"] == "Present":
                flash("Already marked Present.", "error")
                return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

            marked = mark_subject_attendance_present(result.student_id, subject_id, session_date, result.confidence)
            if not marked:
                flash(
                    f"Could not mark attendance for {student['full_name']}. "
                    "Start the subject session first.",
                    "error",
                )
                return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

            flash(
                f"{student['full_name']} marked Present in {subject['subject_name']} with confidence {result.confidence:.1f}%.",
                "success",
            )
        except Exception as exc:
            flash(f"Attendance scan failed: {exc}", "error")

        return redirect(url_for("dashboard", subject_id=subject_id, session_date=session_date))

    return app
