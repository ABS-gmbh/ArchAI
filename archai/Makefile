.PHONY: help backend frontend

help:
	@echo "ArchAI application commands"
	@echo "  make backend    Run the FastAPI backend on 127.0.0.1:8000"
	@echo "  make frontend   Run the Next.js frontend on 127.0.0.1:3001"

backend:
	cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

frontend:
	cd frontend && npm run dev -- --hostname 127.0.0.1 --port 3001
