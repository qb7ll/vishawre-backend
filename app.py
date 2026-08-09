"""
VoiceTrining Backend API
Flask + SQLite Backend for Voice Training System
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import hashlib
import secrets
import tempfile
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.utils import secure_filename

from deepfake_detector import detector_service

app = Flask(__name__)
# تفعيل CORS لجميع المسارات بما فيها الصوت
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/audio/*": {"origins": "*"}
})

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE = os.path.join(BASE_DIR, 'backend', 'database.db')
AUDIO_DIR = os.path.join(BASE_DIR, 'public', 'deepvoice_segments_50')
UPLOAD_DIR = os.path.join(BASE_DIR, 'backend', 'uploads')
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm', '.mp4'}

# Print paths for debugging
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] AUDIO_DIR: {AUDIO_DIR}")

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with all tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'employee' CHECK(role IN ('employee', 'administrator')),
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Voice Samples table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            classification TEXT NOT NULL CHECK(classification IN ('real', 'ai')),
            description TEXT,
            language TEXT DEFAULT 'English',
            duration INTEGER,
            difficulty INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('training', 'assessment')),
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_questions INTEGER,
            correct_answers INTEGER,
            score REAL,
            performance_level TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # Session Responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            sample_id INTEGER NOT NULL,
            user_answer TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            response_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (sample_id) REFERENCES voice_samples(id)
        )
    ''')
    
    # Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Auth tokens table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auth_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def normalize_email(email):
    """Normalize email addresses before lookup or storage"""
    return (email or '').strip().lower()

def generate_token():
    """Generate secure token"""
    return secrets.token_hex(32)


def allowed_audio_file(filename):
    """Check whether the uploaded filename is supported"""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_AUDIO_EXTENSIONS

def token_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN auth_tokens t ON u.id = t.user_id
            WHERE t.token = ? AND t.expires_at > datetime('now')
        ''', (token,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        return f(dict(user), *args, **kwargs)
    return decorated

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated(current_user, *args, **kwargs):
        if current_user['role'] != 'administrator':
            return jsonify({'error': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# ==================== AUTH ROUTES ====================

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user and return token"""
    data = request.json or {}
    email = normalize_email(data.get('email'))
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute('''
        SELECT * FROM users
        WHERE lower(email) = ? AND password_hash = ? AND status = 'active'
    ''', (email, password_hash))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({'error': 'Invalid credentials or account inactive'}), 401
    
    # Generate token
    token = generate_token()
    expires_at = datetime.now() + timedelta(days=7)
    
    cursor.execute('''
        INSERT INTO auth_tokens (user_id, token, expires_at) VALUES (?, ?, ?)
    ''', (user['id'], token, expires_at))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'name': user['name'],
            'role': user['role']
        }
    })

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout user by invalidating token"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM auth_tokens WHERE token = ?', (token,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    """Get current user info"""
    return jsonify({
        'id': current_user['id'],
        'email': current_user['email'],
        'name': current_user['name'],
        'role': current_user['role']
    })


@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """Update the current user profile"""
    data = request.json or {}
    name = (data.get('name') or '').strip()

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET name = ?, updated_at = datetime('now')
        WHERE id = ?
    ''', (name, current_user['id']))
    conn.commit()

    cursor.execute('SELECT id, email, name, role FROM users WHERE id = ?', (current_user['id'],))
    updated_user = cursor.fetchone()
    conn.close()

    return jsonify({
        'success': True,
        'user': dict(updated_user),
    })


@app.route('/api/auth/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    """Change the current user password"""
    data = request.json or {}
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')

    if not current_password or not new_password:
        return jsonify({'error': 'Current and new password are required'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'New password must be at least 6 characters'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT password_hash FROM users WHERE id = ?', (current_user['id'],))
    existing_user = cursor.fetchone()

    if not existing_user or existing_user['password_hash'] != hash_password(current_password):
        conn.close()
        return jsonify({'error': 'Current password is incorrect'}), 400

    cursor.execute('''
        UPDATE users
        SET password_hash = ?, updated_at = datetime('now')
        WHERE id = ?
    ''', (hash_password(new_password), current_user['id']))
    conn.commit()
    conn.close()

    return jsonify({'success': True})

# ==================== VOICE SAMPLES ROUTES ====================

@app.route('/api/samples', methods=['GET'])
@token_required
def get_samples(current_user):
    """Get all voice samples"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM voice_samples ORDER BY created_at DESC')
    samples = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Add full audio URL
    for sample in samples:
        sample['audioUrl'] = f"/audio/{sample['audio_path']}"
    
    return jsonify(samples)

@app.route('/api/samples/<int:sample_id>', methods=['GET'])
@token_required
def get_sample(current_user, sample_id):
    """Get single voice sample"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM voice_samples WHERE id = ?', (sample_id,))
    sample = cursor.fetchone()
    conn.close()
    
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    
    sample_dict = dict(sample)
    sample_dict['audioUrl'] = f"/audio/{sample_dict['audio_path']}"
    return jsonify(sample_dict)

@app.route('/api/samples', methods=['POST'])
@token_required
@admin_required
def add_sample(current_user):
    """Add new voice sample (admin only)"""
    data = request.json
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO voice_samples (name, audio_path, classification, description, language, difficulty)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        data.get('name'),
        data.get('audioPath'),
        data.get('classification'),
        data.get('description', ''),
        data.get('language', 'English'),
        data.get('difficulty', 1)
    ))
    conn.commit()
    sample_id = cursor.lastrowid
    conn.close()
    
    return jsonify({'success': True, 'id': sample_id}), 201

@app.route('/api/samples/<int:sample_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_sample(current_user, sample_id):
    """Delete voice sample (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM voice_samples WHERE id = ?', (sample_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

# ==================== SESSION ROUTES ====================

@app.route('/api/sessions/start', methods=['POST'])
@token_required
def start_session(current_user):
    """Start a new training/assessment session with balanced samples"""
    data = request.json
    mode = data.get('mode', 'training')
    
    # 8 questions total: 4 FAKE + 4 REAL for balance
    half_count = 4
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get 4 random FAKE samples
    cursor.execute('''
        SELECT * FROM voice_samples WHERE classification = 'ai' ORDER BY RANDOM() LIMIT ?
    ''', (half_count,))
    fake_samples = [dict(row) for row in cursor.fetchall()]
    
    # Get 4 random REAL samples
    cursor.execute('''
        SELECT * FROM voice_samples WHERE classification = 'real' ORDER BY RANDOM() LIMIT ?
    ''', (half_count,))
    real_samples = [dict(row) for row in cursor.fetchall()]
    
    # Combine and shuffle
    import random
    all_samples = fake_samples + real_samples
    random.shuffle(all_samples)
    
    # Create session
    cursor.execute('''
        INSERT INTO sessions (user_id, mode, total_questions) VALUES (?, ?, ?)
    ''', (current_user['id'], mode, len(all_samples)))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Prepare samples for frontend (hide classification)
    session_samples = []
    for sample in all_samples:
        session_samples.append({
            'id': sample['id'],
            'name': sample['name'],
            'audioUrl': f"/audio/{sample['audio_path']}",
            'description': sample['description'],
            'language': sample['language'],
            'duration': sample['duration']
        })
    
    return jsonify({
        'sessionId': session_id,
        'mode': mode,
        'samples': session_samples,
        'totalQuestions': len(session_samples)
    })

@app.route('/api/sessions/<int:session_id>/submit', methods=['POST'])
@token_required
def submit_response(current_user, session_id):
    """Submit answer for a sample"""
    data = request.json
    sample_id = data.get('sampleId')
    user_answer = data.get('answer')
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get correct answer
    cursor.execute('SELECT classification FROM voice_samples WHERE id = ?', (sample_id,))
    sample = cursor.fetchone()
    
    if not sample:
        conn.close()
        return jsonify({'error': 'Sample not found'}), 404
    
    correct_answer = sample['classification']
    is_correct = user_answer == correct_answer
    
    # Save response
    cursor.execute('''
        INSERT INTO session_responses (session_id, sample_id, user_answer, correct_answer, is_correct)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, sample_id, user_answer, correct_answer, is_correct))
    conn.commit()
    conn.close()
    
    return jsonify({
        'isCorrect': is_correct,
        'correctAnswer': correct_answer
    })

@app.route('/api/sessions/<int:session_id>/complete', methods=['POST'])
@token_required
def complete_session(current_user, session_id):
    """Complete a session and calculate score"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get all responses for this session
    cursor.execute('''
        SELECT COUNT(*) as total, SUM(is_correct) as correct
        FROM session_responses WHERE session_id = ?
    ''', (session_id,))
    result = cursor.fetchone()
    
    total = result['total']
    correct = result['correct'] or 0
    score = round((correct / total) * 100) if total > 0 else 0
    
    # Determine performance level
    if score >= 90:
        performance_level = 'Excellent'
    elif score >= 75:
        performance_level = 'Good'
    elif score >= 60:
        performance_level = 'Satisfactory'
    elif score >= 40:
        performance_level = 'Needs Improvement'
    else:
        performance_level = 'Poor'
    
    # Update session
    cursor.execute('''
        UPDATE sessions SET 
            end_time = datetime('now'),
            correct_answers = ?,
            score = ?,
            performance_level = ?
        WHERE id = ?
    ''', (correct, score, performance_level, session_id))
    conn.commit()
    
    # Get all responses with sample info
    cursor.execute('''
        SELECT sr.*, vs.name as sample_name
        FROM session_responses sr
        JOIN voice_samples vs ON sr.sample_id = vs.id
        WHERE sr.session_id = ?
    ''', (session_id,))
    responses = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({
        'sessionId': session_id,
        'totalQuestions': total,
        'correctAnswers': correct,
        'incorrectAnswers': total - correct,
        'score': score,
        'performanceLevel': performance_level,
        'responses': responses
    })

@app.route('/api/sessions/history', methods=['GET'])
@token_required
def get_session_history(current_user):
    """Get user's session history"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM sessions 
        WHERE user_id = ? AND end_time IS NOT NULL
        ORDER BY end_time DESC
        LIMIT 50
    ''', (current_user['id'],))
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(sessions)

# ==================== ADMIN ROUTES ====================

@app.route('/api/admin/users', methods=['GET'])
@token_required
@admin_required
def get_all_users(current_user):
    """Get all users (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, email, name, role, status, created_at FROM users ORDER BY created_at DESC')
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(users)

@app.route('/api/admin/users', methods=['POST'])
@token_required
@admin_required
def create_user(current_user):
    """Create new user (admin only)"""
    data = request.json or {}
    email = normalize_email(data.get('email'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (email, password_hash, name, role, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            email,
            hash_password(data.get('password')),
            data.get('name'),
            data.get('role', 'employee'),
            data.get('status', 'active')
        ))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return jsonify({'success': True, 'id': user_id}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Email already exists'}), 400

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@token_required
@admin_required
def update_user(current_user, user_id):
    """Update user (admin only)"""
    data = request.json or {}
    
    conn = get_db()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if 'name' in data:
        updates.append('name = ?')
        params.append(data['name'])
    if 'email' in data:
        updates.append('email = ?')
        params.append(normalize_email(data['email']))
    if 'role' in data:
        updates.append('role = ?')
        params.append(data['role'])
    if 'status' in data:
        updates.append('status = ?')
        params.append(data['status'])
    if 'password' in data:
        updates.append('password_hash = ?')
        params.append(hash_password(data['password']))
    
    params.append(user_id)
    
    cursor.execute(f'''
        UPDATE users SET {', '.join(updates)}, updated_at = datetime('now')
        WHERE id = ?
    ''', params)
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@token_required
@admin_required
def delete_user(current_user, user_id):
    """Delete user (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/admin/reports', methods=['GET'])
@token_required
@admin_required
def get_reports(current_user):
    """Get performance reports (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get employee stats
    cursor.execute('''
        SELECT 
            u.id, u.name, u.email,
            COUNT(s.id) as total_sessions,
            COALESCE(AVG(s.score), 0) as average_score,
            MAX(s.end_time) as last_session
        FROM users u
        LEFT JOIN sessions s ON u.id = s.user_id AND s.end_time IS NOT NULL
        WHERE u.role = 'employee'
        GROUP BY u.id
    ''')
    employees = [dict(row) for row in cursor.fetchall()]
    
    # Get overall stats
    cursor.execute('''
        SELECT 
            COUNT(DISTINCT user_id) as active_users,
            COUNT(*) as total_sessions,
            COALESCE(AVG(score), 0) as overall_average
        FROM sessions WHERE end_time IS NOT NULL
    ''')
    overall = dict(cursor.fetchone())
    
    conn.close()
    
    return jsonify({
        'employees': employees,
        'overall': overall
    })

@app.route('/api/admin/settings', methods=['GET'])
@token_required
@admin_required
def get_settings(current_user):
    """Get all settings (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT key, value FROM settings')
    settings = {row['key']: row['value'] for row in cursor.fetchall()}
    conn.close()
    
    # Default settings
    defaults = {
        'samplesPerSession': '10',
        'selectionRule': 'random',
        'balanceRatio': '50',
        'showFeedbackInTraining': 'true',
        'allowReplay': 'true',
        'minPlaybackPercent': '80'
    }
    
    return jsonify({**defaults, **settings})

@app.route('/api/admin/settings', methods=['PUT'])
@token_required
@admin_required
def update_settings(current_user):
    """Update settings (admin only)"""
    data = request.json
    
    conn = get_db()
    cursor = conn.cursor()
    
    for key, value in data.items():
        cursor.execute('''
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
        ''', (key, str(value)))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})


# ==================== DETECTOR ROUTES ====================

@app.route('/api/detector/status', methods=['GET'])
@token_required
def get_detector_status(current_user):
    """Get deepfake detector model status and metadata"""
    model_id = request.args.get('model_id')
    metadata = detector_service.get_metadata(model_id)
    
    from deepfake_detector import AVAILABLE_MODELS
    available_models_list = [
        {"id": m_id, "type": config["type"]}
        for m_id, config in AVAILABLE_MODELS.items()
    ]
    
    return jsonify({
        'modelId': metadata.model_id,
        'localPath': metadata.model_dir,
        'downloaded': metadata.downloaded,
        'ready': metadata.ready,
        'device': metadata.device,
        'message': metadata.message,
        'supportedFormats': sorted(ALLOWED_AUDIO_EXTENSIONS),
        'availableModels': available_models_list
    })


@app.route('/api/detector/analyze', methods=['POST'])
@token_required
def analyze_audio(current_user):
    """Analyze uploaded audio and classify it as real or AI-generated"""
    if 'audio' not in request.files:
        return jsonify({'error': 'Audio file is required'}), 400

    audio_file = request.files['audio']
    if not audio_file or not audio_file.filename:
        return jsonify({'error': 'Audio file is required'}), 400

    filename = secure_filename(audio_file.filename)
    if not allowed_audio_file(filename):
        return jsonify({'error': 'Unsupported audio format'}), 400

    model_id = request.form.get('model_id')

    import tempfile
    import uuid
    _, ext = os.path.splitext(filename)
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")

    try:
        audio_file.save(temp_path)
        result = detector_service.analyze(temp_path, model_id=model_id)
        return jsonify({
            'success': True,
            'filename': filename,
            **result,
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        print(f"[ERROR] Detector failed: {exc}")
        return jsonify({'error': 'Failed to analyze audio'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.route('/api/detector/tts', methods=['POST'])
@token_required
def text_to_speech(current_user):
    """Generate audio from text locally for deepfake testing"""
    import tempfile
    import uuid
    from flask import after_this_request, send_file
    import pyttsx3

    data = request.json or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    temp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        engine.stop()

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Failed to generate audio - file is empty'}), 500

        @after_this_request
        def cleanup_temp_file(response):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError as cleanup_error:
                print(f"[WARN] Failed to remove TTS temp file: {cleanup_error}")
            return response

        return send_file(
            temp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_speech.wav'
        )
    except Exception as exc:
        print(f"[ERROR] TTS failed: {exc}")
        return jsonify({'error': f'TTS engine failure: {exc}'}), 500
        # ==================== AUDIO FILES ====================

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files"""
    # Determine directory based on filename
    if filename.startswith('FAKE'):
        directory = os.path.join(AUDIO_DIR, 'FAKE')
    elif filename.startswith('REAL'):
        directory = os.path.join(AUDIO_DIR, 'REAL')
    else:
        return jsonify({'error': 'Invalid audio path'}), 404
    
    return send_from_directory(directory, filename)

# ==================== INITIALIZATION ====================

def seed_initial_data():
    """Seed initial data into database"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) as count FROM users')
    if cursor.fetchone()['count'] > 0:
        conn.close()
        return
    
    # Add default users
    users = [
        ('employee@voice.com', hash_password('employee123'), 'Fatima', 'employee', 'active'),
        ('admin@voice.com', hash_password('admin123'), 'Admin', 'administrator', 'active'),
        ('mayar@voice.com', hash_password('admin123'), 'Mayar', 'administrator', 'active'),
        ('test@voice.com', hash_password('test123'), 'Razan', 'employee', 'active'),
    ]
    
    cursor.executemany('''
        INSERT INTO users (email, password_hash, name, role, status)
        VALUES (?, ?, ?, ?, ?)
    ''', users)
    
    # Add voice samples from FAKE and REAL directories
    samples = []
    
    # FAKE samples
    for i in range(1, 51):
        samples.append((
            f'AI Voice Sample {i}',
            f'FAKE_seg_{i}.wav',
            'ai',
            f'AI-generated voice sample #{i} - Deepfake audio',
            'English',
            6,  # duration in seconds (approximate)
            1 + (i % 3)  # difficulty 1-3
        ))
    
    # REAL samples
    for i in range(1, 51):
        samples.append((
            f'Human Voice Sample {i}',
            f'REAL_seg_{i}.wav',
            'real',
            f'Real human voice sample #{i} - Authentic audio',
            'English',
            7,  # duration in seconds (approximate)
            1 + (i % 3)  # difficulty 1-3
        ))
    
    cursor.executemany('''
        INSERT INTO voice_samples (name, audio_path, classification, description, language, duration, difficulty)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', samples)
    
    # Add default settings
    settings = [
        ('samplesPerSession', '10'),
        ('selectionRule', 'random'),
        ('balanceRatio', '50'),
        ('showFeedbackInTraining', 'true'),
        ('allowReplay', 'true'),
        ('minPlaybackPercent', '80'),
    ]
    
    cursor.executemany('''
        INSERT INTO settings (key, value) VALUES (?, ?)
    ''', settings)
    
    conn.commit()
    conn.close()
    print("[OK] Database seeded with initial data!")

def ensure_core_users():
    """Ensure required demo accounts exist in existing databases"""
    conn = get_db()
    cursor = conn.cursor()
    users = [
        ('employee@voice.com', hash_password('employee123'), 'Fatima', 'employee', 'active'),
        ('admin@voice.com', hash_password('admin123'), 'Admin', 'administrator', 'active'),
        ('mayar@voice.com', hash_password('admin123'), 'Mayar', 'administrator', 'active'),
        ('test@voice.com', hash_password('test123'), 'Razan', 'employee', 'active'),
    ]

    for email, password_hash, name, role, status in users:
        normalized_email = normalize_email(email)
        cursor.execute('SELECT id FROM users WHERE lower(email) = ?', (normalized_email,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.execute('''
                UPDATE users
                SET email = ?, role = ?, status = ?, updated_at = datetime('now')
                WHERE id = ?
            ''', (normalized_email, role, status, existing_user['id']))
        else:
            cursor.execute('''
                INSERT INTO users (email, password_hash, name, role, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (normalized_email, password_hash, name, role, status))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    print("[*] Initializing VoiceTrining Backend...")
    init_db()
    seed_initial_data()
    ensure_core_users()
    print("[OK] Database initialized!")
    print("[*] Starting server on http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)

