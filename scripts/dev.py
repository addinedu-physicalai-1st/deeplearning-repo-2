import os
import signal
import subprocess
import sys
import time
import socket
import shutil
import threading
from pathlib import Path

# --- Configuration ---
# Format: { name: { "path": path, "port": port, "cmd": cmd } }
APPS = {
    "operation_server": {
        "path": "apps/operation_server",
        "port": 8000,
        "cmd": ["uv", "run", "python", "src/operation_server/main.py"]
    },
    "ai_head": {
        "path": "apps/ai_head",
        "port": 8001,
        "cmd": ["uv", "run", "python", "src/ai_head/main.py"]
    },
    "ai_emotion": {
        "path": "apps/ai_emotion",
        "port": 8002,
        "cmd": ["uv", "run", "python", "src/ai_emotion/main.py"]
    },
    "ai_body": {
        "path": "apps/ai_body",
        "port": 8003,
        "cmd": ["uv", "run", "python", "src/ai_body/main.py"]
    },
    "llm_server": {
        "path": "apps/llm_server",
        "port": 8004,
        "cmd": ["uv", "run", "python", "src/llm_server/main.py"]
    },
    "ai_interface": {
        "path": "apps/ai_interface",
        "port": 8010,
        "cmd": ["uv", "run", "python", "src/ai_interface/main.py"]
    },
}

# 필수 모델 파일 정의 (체크용)
REQUIRED_FILES = {
    "ai_emotion": ["best.pt"],
}

# format: { name: { "path": path, "cmd": cmd } }
CLIENT_APPS = {
    "client": {
        "path": "apps/client",
        "cmd": ["uv", "run", "python", "src/client/main.py"]
    }
}

# Colors for terminal output
COLORS = [
    "\033[92m", # Green
    "\033[94m", # Blue
    "\033[93m", # Yellow
    "\033[95m", # Magenta
    "\033[96m", # Cyan
    "\033[91m", # Red
]
RESET = "\033[0m"

def check_port(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def validate_env(app_name, app_path):
    """Validate .env file existence and basic content."""
    env_path = Path(app_path) / ".env"
    example_path = Path(app_path) / ".env.example"
    
    if not env_path.exists():
        if example_path.exists():
            print(f"⚠️  [{app_name}] .env 파일이 없습니다. .env.example에서 복사합니다...")
            shutil.copy(example_path, env_path)
            return True, "Created from .env.example"
        else:
            return False, ".env 파일과 .env.example 파일이 모두 없습니다."
    
    # Check for API_KEY
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            has_api_key = False
            for line in lines:
                if line.strip().startswith("API_KEY="):
                    value = line.split("=", 1)[1].strip().strip("'").strip('"')
                    if value:
                        has_api_key = True
                        break
            
            if not has_api_key:
                return False, "API_KEY가 설정되지 않았거나 비어있습니다. .env 파일을 확인해주세요."
    except Exception as e:
        return False, f"파일 읽기 오류: {str(e)}"
            
    return True, "OK"

def log_relay(name, process, color):
    """Relay process output to console with prefix and color."""
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"{color}[{name}]{RESET} {line.strip()}")
    process.stdout.close()

def main():
    processes = {}
    threads = []
    
    print("=" * 60)
    print("🚀 Focus Monitor 통합 개발 서버 실행기")
    print("=" * 60)
    
    # 1. 사전 점검
    print("\n🔍 사전 점검 중...")
    errors = []
    for name, config in APPS.items():
        # 포트 점검
        if check_port(config["port"]):
            errors.append(f"❌ 포트 {config['port']}가 이미 사용 중입니다 ({name} 실행 불가)")
        
        # 환경 변수 점검
        valid, msg = validate_env(name, config["path"])
        if not valid:
            errors.append(f"❌ [{name}] .env 오류: {msg}")
            
        # 필수 파일 점검 (예: best.pt)
        if name in REQUIRED_FILES:
            app_path = Path(config["path"])
            for file_name in REQUIRED_FILES[name]:
                if not (app_path / file_name).exists():
                    errors.append(f"❌ [{name}] 필수 파일 누락: {file_name}\n   👉 해당 파일을 {app_path}/ 폴더에 넣어주세요.")
            
    if errors:
        print("\n".join(errors))
        print("\n🛑 오류가 발견되어 실행을 중단합니다.")
        sys.exit(1)
        
    print("✅ 모든 점검을 통과했습니다.")

    # 2. 프로세스 시작
    try:
        root_dir = Path.cwd()
        for i, (name, config) in enumerate(APPS.items()):
            color = COLORS[i % len(COLORS)]
            print(f"📦 {name} 시작 중... (Port: {config['port']})")
            
            cwd = root_dir / config["path"]
            
            # subprocess.Popen으로 실행
            p = subprocess.Popen(
                config["cmd"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            processes[name] = p
            
            # 로그 릴레이용 스레드 시작
            t = threading.Thread(target=log_relay, args=(name, p, color), daemon=True)
            t.start()
            threads.append(t)
            
            # 서버 간 시작 간격 (선택 사항)
            time.sleep(0.5)
            
        print("\n" + "=" * 60)
        print("✅ 모든 서버가 실행되었습니다.")
        
        # 3. 클라이언트 프로그램 시작
        for name, config in CLIENT_APPS.items():
            color = COLORS[(len(APPS) + list(CLIENT_APPS.keys()).index(name)) % len(COLORS)]
            print(f"🖥️  {name} 시작 중...")
            cwd = root_dir / config["path"]
            p = subprocess.Popen(
                config["cmd"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            processes[name] = p
            t = threading.Thread(target=log_relay, args=(name, p, color), daemon=True)
            t.start()
            threads.append(t)

        print("💡 종료하려면 Ctrl+C를 누르세요.")
        print("=" * 60 + "\n")
        
        # 메인 루프: 프로세스 상태 감시
        while True:
            for name, p in processes.items():
                if p.poll() is not None:
                    print(f"\n🛑 {name} 프로세스가 예기치 않게 종료되었습니다 (Exit Code: {p.returncode})")
                    raise KeyboardInterrupt
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 종료 요청을 받았습니다. 모든 서버를 종료합니다...")
    finally:
        # 3. 모든 프로세스 종료
        for name, p in processes.items():
            if p.poll() is None:
                print(f"🧹 {name} 종료 중...")
                # 프로세스 그룹 전체를 종료하려면 os.killpg가 필요할 수 있으나 
                # 여기서는 단순 terminate로 시작
                p.terminate()
        
        # 종료 대기
        for name, p in processes.items():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️  {name}가 정상적으로 종료되지 않아 강제 종료합니다.")
                p.kill()
        
        print("\n✨ 모든 서버가 성공적으로 종료되었습니다.")

def stop_servers():
    """Kill any lingering processes on the configured ports."""
    print("🧹 기존 서버 프로세스 종료 중...")
    import platform
    
    current_os = platform.system()
    
    for name, config in APPS.items():
        port = config["port"]
        try:
            if current_os == "Windows":
                # Windows: netstat을 사용하여 PID 찾기
                cmd = f"netstat -ano | findstr :{port}"
                output = subprocess.check_output(cmd, shell=True, text=True)
                for line in output.strip().split("\n"):
                    if "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        print(f"  - {name} (Port {port}, PID {pid}) 종료 중...")
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            else:
                # macOS/Linux: lsof 사용
                try:
                    result = subprocess.check_output(["lsof", "-t", f"-i:{port}"], text=True)
                    pids = result.strip().split("\n")
                    for pid in pids:
                        if pid:
                            print(f"  - {name} (Port {port}, PID {pid}) 종료 중...")
                            os.kill(int(pid), signal.SIGTERM)
                except subprocess.CalledProcessError:
                    pass
        except Exception:
            # 포트를 사용 중인 프로세스가 없으면 무시
            pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_servers()
    else:
        main()
