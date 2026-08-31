.PHONY: help backend-install backend-dev frontend-install frontend-dev vendor-backend vendor-frontend

help:
	@echo "ArchAI workspace commands"
	@echo "  make backend-install   Install prototype backend in editable mode"
	@echo "  make backend-dev       Run MOCK prototype backend (fabricated output)"
	@echo "  make frontend-install  Install prototype frontend dependencies"
	@echo "  make frontend-dev      Run prototype frontend (Vite)"
	@echo "  make vendor-backend    Run vendor layout backend on 127.0.0.1:8000"
	@echo "  make vendor-frontend   Run vendor layout frontend (Next.js)"

backend-install:
	cd backend && python -m pip install -e .

backend-dev:
	@echo "WARNING: backend/ is MOCK scaffolding. /ingest/image returns fabricated"
	@echo "         span text, not OCR output, and is disabled unless"
	@echo "         ARCHAI_ALLOW_MOCK_BACKEND=1 is set."
	@echo "         For the real OCR service use: make vendor-backend"
	@echo ""
	cd backend && uvicorn archai_backend.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

vendor-backend:
	cd vendor/layout/backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

vendor-frontend:
	cd vendor/layout/frontend && npm run dev
