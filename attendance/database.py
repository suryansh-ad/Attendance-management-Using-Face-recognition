import sqlite3
from contextlib import contextmanager

from .config import DATA_DIR, DB_PATH


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    ensure_data_dir()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                department TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                marked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                marked_on TEXT NOT NULL,
                confidence REAL NOT NULL,
                UNIQUE(student_id, marked_on),
                FOREIGN KEY(student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_code TEXT NOT NULL UNIQUE,
                subject_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS subject_enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, subject_id),
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS subject_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                session_date TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                marked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, subject_id, session_date),
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            );
            """
        )
        connection.commit()


def fetch_all_students() -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, student_code, full_name, department, created_at
            FROM students
            ORDER BY full_name ASC
            """
        ).fetchall()
    return list(rows)


def fetch_student(student_id: int) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, student_code, full_name, department, created_at
            FROM students
            WHERE id = ?
            """,
            (student_id,),
        ).fetchone()
    return row


def fetch_student_by_code(student_code: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, student_code, full_name, department, created_at
            FROM students
            WHERE student_code = ?
            """,
            (student_code.strip(),),
        ).fetchone()
    return row


def fetch_student_by_identity(full_name: str, department: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, student_code, full_name, department, created_at
            FROM students
            WHERE LOWER(TRIM(full_name)) = LOWER(TRIM(?))
              AND LOWER(TRIM(department)) = LOWER(TRIM(?))
            """,
            (full_name, department),
        ).fetchone()
    return row


def create_student(student_code: str, full_name: str, department: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO students (student_code, full_name, department)
            VALUES (?, ?, ?)
            """,
            (student_code.strip(), full_name.strip(), department.strip()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def delete_student(student_id: int) -> bool:
    with get_connection() as connection:
        student_deleted = connection.execute(
            "DELETE FROM students WHERE id = ?",
            (student_id,),
        )
        connection.commit()
    return bool(student_deleted.rowcount)


def fetch_all_subjects() -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, subject_code, subject_name, created_at
            FROM subjects
            ORDER BY subject_name ASC
            """
        ).fetchall()
    return list(rows)


def fetch_subject(subject_id: int) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, subject_code, subject_name, created_at
            FROM subjects
            WHERE id = ?
            """,
            (subject_id,),
        ).fetchone()
    return row


def fetch_subject_by_code(subject_code: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, subject_code, subject_name, created_at
            FROM subjects
            WHERE subject_code = ?
            """,
            (subject_code.strip(),),
        ).fetchone()
    return row


def create_subject(subject_code: str, subject_name: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO subjects (subject_code, subject_name)
            VALUES (?, ?)
            """,
            (subject_code.strip(), subject_name.strip()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def delete_subject(subject_id: int) -> bool:
    with get_connection() as connection:
        deleted = connection.execute(
            "DELETE FROM subjects WHERE id = ?",
            (subject_id,),
        )
        connection.commit()
    return bool(deleted.rowcount)


def subject_count() -> int:
    with get_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM subjects").fetchone()
    return int(row["count"])


def enroll_student_in_subject(student_id: int, subject_id: int) -> bool:
    with get_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO subject_enrollments (student_id, subject_id)
                VALUES (?, ?)
                """,
                (student_id, subject_id),
            )
            connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def remove_student_from_subject(student_id: int, subject_id: int) -> bool:
    with get_connection() as connection:
        deleted = connection.execute(
            """
            DELETE FROM subject_enrollments
            WHERE student_id = ? AND subject_id = ?
            """,
            (student_id, subject_id),
        )
        connection.commit()
    return bool(deleted.rowcount)


def is_student_enrolled_in_subject(student_id: int, subject_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM subject_enrollments
            WHERE student_id = ? AND subject_id = ?
            """,
            (student_id, subject_id),
        ).fetchone()
    return row is not None


def fetch_subject_enrollments(subject_id: int) -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT students.id, students.student_code, students.full_name, students.department
            FROM subject_enrollments
            INNER JOIN students ON students.id = subject_enrollments.student_id
            WHERE subject_enrollments.subject_id = ?
            ORDER BY students.full_name ASC
            """,
            (subject_id,),
        ).fetchall()
    return list(rows)


def enrolled_student_count(subject_id: int) -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM subject_enrollments WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
    return int(row["count"])


def initialize_subject_attendance(subject_id: int, session_date: str) -> int:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO subject_attendance (student_id, subject_id, status, session_date, confidence)
            SELECT student_id, subject_id, 'Absent', ?, 0
            FROM subject_enrollments
            WHERE subject_id = ?
            """,
            (session_date, subject_id),
        )
        connection.commit()
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM subject_attendance
            WHERE subject_id = ? AND session_date = ?
            """,
            (subject_id, session_date),
        ).fetchone()
    return int(row["count"])


def mark_subject_attendance_present(student_id: int, subject_id: int, session_date: str, confidence: float) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE subject_attendance
            SET status = 'Present',
                confidence = ?,
                marked_at = CURRENT_TIMESTAMP
            WHERE student_id = ? AND subject_id = ? AND session_date = ?
            """,
            (confidence, student_id, subject_id, session_date),
        )
        connection.commit()
    return bool(cursor.rowcount)


def fetch_subject_attendance_record(student_id: int, subject_id: int, session_date: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, student_id, subject_id, status, session_date, confidence, marked_at
            FROM subject_attendance
            WHERE student_id = ? AND subject_id = ? AND session_date = ?
            """,
            (student_id, subject_id, session_date),
        ).fetchone()
    return row


def fetch_subject_attendance_logs(subject_id: int | None = None, session_date: str | None = None, limit: int = 300) -> list[sqlite3.Row]:
    return fetch_subject_attendance_logs_by_status(subject_id, session_date, None, limit)


def fetch_subject_attendance_logs_by_status(
    subject_id: int | None = None,
    session_date: str | None = None,
    status: str | None = None,
    limit: int = 300,
) -> list[sqlite3.Row]:
    query = """
        SELECT
            subject_attendance.id,
            subject_attendance.subject_id,
            subjects.subject_code,
            subjects.subject_name,
            students.student_code,
            students.full_name,
            students.department,
            subject_attendance.status,
            subject_attendance.confidence,
            subject_attendance.session_date,
            subject_attendance.marked_at
        FROM subject_attendance
        INNER JOIN students ON students.id = subject_attendance.student_id
        INNER JOIN subjects ON subjects.id = subject_attendance.subject_id
        WHERE 1 = 1
    """
    params: list[object] = []
    if subject_id is not None:
        query += " AND subject_attendance.subject_id = ?"
        params.append(subject_id)
    if session_date is not None:
        query += " AND subject_attendance.session_date = ?"
        params.append(session_date)
    if status in {"Present", "Absent"}:
        query += " AND subject_attendance.status = ?"
        params.append(status)
    query += " ORDER BY subject_attendance.session_date DESC, subject_attendance.marked_at DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return list(rows)


def subject_attendance_summary(subject_id: int, session_date: str) -> dict[str, int]:
    with get_connection() as connection:
        present_row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM subject_attendance
            WHERE subject_id = ? AND session_date = ? AND status = 'Present'
            """,
            (subject_id, session_date),
        ).fetchone()
        absent_row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM subject_attendance
            WHERE subject_id = ? AND session_date = ? AND status = 'Absent'
            """,
            (subject_id, session_date),
        ).fetchone()
    return {
        "present": int(present_row["count"]),
        "absent": int(absent_row["count"]),
    }


def fetch_subject_roster_statuses(subject_id: int, session_date: str) -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                students.id,
                students.student_code,
                students.full_name,
                students.department,
                CASE
                    WHEN subject_attendance.status IS NULL THEN 'Absent'
                    ELSE subject_attendance.status
                END AS attendance_status,
                COALESCE(subject_attendance.confidence, 0) AS confidence
            FROM subject_enrollments
            INNER JOIN students
                ON students.id = subject_enrollments.student_id
            LEFT JOIN subject_attendance
                ON subject_attendance.student_id = students.id
                AND subject_attendance.subject_id = ?
                AND subject_attendance.session_date = ?
            WHERE subject_enrollments.subject_id = ?
            ORDER BY students.full_name ASC
            """,
            (subject_id, session_date, subject_id),
        ).fetchall()
    return list(rows)


def fetch_attendance_logs(limit: int = 200) -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                attendance_logs.id,
                students.student_code,
                students.full_name,
                students.department,
                attendance_logs.status,
                attendance_logs.marked_at,
                attendance_logs.marked_on,
                attendance_logs.confidence
            FROM attendance_logs
            INNER JOIN students ON students.id = attendance_logs.student_id
            ORDER BY attendance_logs.marked_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return list(rows)


def has_attendance_for_day(student_id: int, marked_on: str) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM attendance_logs
            WHERE student_id = ? AND marked_on = ?
            """,
            (student_id, marked_on),
        ).fetchone()
    return row is not None


def mark_attendance(student_id: int, marked_on: str, confidence: float) -> bool:
    with get_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO attendance_logs (student_id, marked_on, confidence)
                VALUES (?, ?, ?)
                """,
                (student_id, marked_on, confidence),
            )
            connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def student_count() -> int:
    with get_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM students").fetchone()
    return int(row["count"])


def attendance_count_today(marked_on: str) -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM attendance_logs WHERE marked_on = ?",
            (marked_on,),
        ).fetchone()
    return int(row["count"])
