import sqlite3
import json
import os
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_name=None):
        if db_name is None:
            # 기본 경로를 app/data/focus_log.db로 설정
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            self.db_name = os.path.join(data_dir, "focus_log.db")
        else:
            self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS focus_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                start_time TEXT NOT NULL,
                end_time TEXT,
                model_type TEXT NOT NULL,
                duration INTEGER,
                focus_score INTEGER,
                distract_cnt INTEGER,
                log_json TEXT,
                ai_feedback TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def insert_session(self, data: dict):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO focus_sessions 
            (start_time, end_time, model_type, duration, focus_score, distract_cnt, log_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['start_time'],
            data['end_time'],
            data['model_type'],
            data['duration'],
            data['focus_score'],
            data['distract_cnt'],
            json.dumps(data.get('log_json', []))
        ))
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id

    def update_feedback(self, session_id, feedback_text):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('UPDATE focus_sessions SET ai_feedback = ? WHERE id = ?', (feedback_text, session_id))
        conn.commit()
        conn.close()

    def delete_session(self, session_id):
        """[Fix 4] 특정 세션 삭제 기능 추가"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM focus_sessions WHERE id = ?', (session_id,))
        conn.commit()
        conn.close()

    def get_all_sessions(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM focus_sessions ORDER BY start_time DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
