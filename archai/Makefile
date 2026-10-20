.PHONY: help frontend-install frontend-dev vendor-backend vendor-frontend

help:
	@echo "ArchAI workspace commands"
	@echo "  make frontend-install  Install prototype frontend dependencies"
	@echo "  make frontend-dev      Run prototype frontend (Vite)"
	@echo "  make vendor-backend    Run vendor layout backend on 127.0.0.1:8000"
	@echo "  make vendor-frontend   Run vendor layout frontend (Next.js)"

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

vendor-backend:
	cd vendor/layout/backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

vendor-frontend:
	cd vendor/layout/frontend && npm run dev
