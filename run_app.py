#!/usr/bin/env python
"""
EcoSort Full-Stack App Runner
=============================
Starts the FastAPI backend and web interface at http://localhost:8000
"""
import sys
import os
import argparse
import uvicorn

if __name__ == '__main__':
    default_host = "0.0.0.0" if ("PORT" in os.environ or "SPACE_ID" in os.environ) else "127.0.0.1"
    default_port = int(os.environ.get("PORT", 7860 if "SPACE_ID" in os.environ else 8000))

    parser = argparse.ArgumentParser(description="Run EcoSort Web Application")
    parser.add_argument("--host", type=str, default=default_host, help="Host IP address (0.0.0.0 for public access)")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind server")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for development")
    args = parser.parse_args()

    print("=" * 65)
    print("🌿 EcoSort: Real-time Edge Waste Classifier")
    print(f"🚀 Server running at: http://{args.host}:{args.port}")
    print(f"📚 Interactive Swagger API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 65)

    uvicorn.run("web_app.server:app", host=args.host, port=args.port, reload=args.reload)
