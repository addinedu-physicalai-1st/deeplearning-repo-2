# ==============================================================================
# [Monitor Project Execution]
# 각 컴퓨터의 역할에 맞는 명령어를 실행하세요.
# 예: 컴퓨터 A에서 오케스트레이터 실행 -> make run-orchestrator
# ==============================================================================

# 1. [Server] Orchestrator (Main Controller) - Port 8000
run-orchestrator:
	@echo "🚀 Starting Orchestrator Server..."
	cd apps/orchestrator && uv run uvicorn src.orchestrator.main:app --host 0.0.0.0 --port 8000 --reload

# 2. [AI] Head Pose Model - Port 8001
run-ai-head:
	@echo "👤 Starting Head Pose Server..."
	cd apps/ai_head && uv run uvicorn src.ai_head.main:app --host 0.0.0.0 --port 8001 --reload

# 3. [AI] Emotion Model - Port 8002
run-ai-emotion:
	@echo "😊 Starting Emotion Server..."
	cd apps/ai_emotion && uv run uvicorn src.ai_emotion.main:app --host 0.0.0.0 --port 8002 --reload

# 4. [AI] Upper Body Model - Port 8003
run-ai-body:
	@echo "💪 Starting Upper Body Server..."
	cd apps/ai_body && uv run uvicorn src.ai_body.main:app --host 0.0.0.0 --port 8003 --reload

# 5. [AI] LLM Server - Port 8004
run-llm:
	@echo "🤖 Starting LLM Server..."
	cd apps/llm_server && uv run uvicorn src.llm_server.main:app --host 0.0.0.0 --port 8004 --reload

# 6. [DB] Database Server (SQLite Wrapper) - Port 8005
run-db:
	@echo "💾 Starting DB API Server..."
	cd apps/db_server && uv run uvicorn src.db_server.main:app --host 0.0.0.0 --port 8005 --reload

# 7. [Client] User Application
run-client:
	@echo "🖥️ Starting Client App..."
	cd apps/client && uv run python src/client/main.py
