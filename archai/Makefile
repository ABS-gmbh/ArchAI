.PHONY: help vendor-backend vendor-frontend

help:
	@echo "ArchAI workspace commands"
	@echo "  make vendor-backend    Run vendor layout backend on 127.0.0.1:8000"
	@echo "  make vendor-frontend   Run vendor layout frontend (Next.js)"

vendor-backend:
	cd vendor/layout/backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

vendor-frontend:
	cd vendor/layout/frontend && npm run dev
