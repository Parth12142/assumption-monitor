#!/usr/bin/env python3
"""Web UI & API server for the Assumption Monitor Tesla-HMI + Three.js Demo (`demo/app.py`).

Run:
    python3 demo/app.py --port 8080
or:
    python3 demo/app.py --cli
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
STATIC_DIR = DEMO_DIR / "static"

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from monitor_engine import GENERATED_PDF_DIR, DemoEngine

ENGINE: DemoEngine | None = None


def get_engine() -> DemoEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = DemoEngine(ROOT_DIR)
    return ENGINE


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serves the interactive single-page UI, Three.js vendor bundle, and JSON API endpoints."""

    def log_message(self, fmt: str, *args):
        sys.stderr.write(f"[AssumptionMonitor Demo] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK, head_only: bool = False):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_file(self, file_path: Path, download_name: str | None = None, head_only: bool = False):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_HEAD(self):
        self._handle_get_or_head(head_only=True)

    def do_GET(self):
        self._handle_get_or_head(head_only=False)

    def _handle_get_or_head(self, head_only: bool = False):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            engine = get_engine()
            if route in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", head_only=head_only)
                return
            if route.startswith("/static/"):
                rel = route[len("/static/"):]
                target = (STATIC_DIR / rel).resolve()
                if STATIC_DIR.resolve() in target.parents or target == STATIC_DIR.resolve():
                    self._send_file(target, head_only=head_only)
                    return
                self.send_error(HTTPStatus.FORBIDDEN)
                return

            if route == "/api/health":
                self._send_json({"status": "ok", "service": "assumption-monitor-demo"}, head_only=head_only)
                return

            if route == "/api/download-pdf":
                doc = query.get("doc", ["plan3"])[0]
                if doc == "plan1":
                    self._send_file(ROOT_DIR / "data" / "raw" / "plan1.pdf", "plan1.pdf", head_only=head_only)
                elif doc == "plan2":
                    self._send_file(ROOT_DIR / "data" / "raw" / "plan2.docx", "plan2.docx", head_only=head_only)
                else:
                    gen_path = GENERATED_PDF_DIR / "plan3.pdf"
                    if not gen_path.exists():
                        engine.generate_sample_pdf()
                    self._send_file(gen_path, "plan3.pdf", head_only=head_only)
                return

            if route == "/api/series":
                years = int(query.get("years", ["3"])[0])
                self._send_json(engine.get_series_explorer_data(years=years), head_only=head_only)
                return

            if route == "/api/model-lab":
                self._send_json(engine.get_model_lab_data(retrain=False), head_only=head_only)
                return

            if route == "/api/analyze":
                doc = query.get("doc", ["plan1"])[0]
                as_of = query.get("as_of", ["2026-10-16"])[0]
                mode = query.get("mode", ["hybrid"])[0]
                threshold = float(query.get("threshold", ["0.45"])[0])
                self._send_json(
                    engine.analyze_document(
                        doc_source=doc,
                        as_of=as_of,
                        mode=mode,
                        threshold=threshold,
                    ),
                    head_only=head_only,
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {route}")
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR, head_only=head_only)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        try:
            engine = get_engine()
            body = self._read_json_body()

            if route == "/api/analyze":
                doc_source = body.get("doc_source", "plan1")
                as_of = body.get("as_of", "2026-10-16")
                mode = body.get("mode", "hybrid")
                threshold = float(body.get("threshold", 0.45))
                override_oil = body.get("override_oil")
                override_fx = body.get("override_fx")
                custom_text = body.get("custom_text")
                uploaded_b64 = body.get("uploaded_b64")
                uploaded_filename = body.get("uploaded_filename")

                uploaded_bytes = base64.b64decode(uploaded_b64) if uploaded_b64 else None
                result = engine.analyze_document(
                    doc_source=doc_source,
                    as_of=as_of,
                    mode=mode,
                    threshold=threshold,
                    override_oil=float(override_oil) if override_oil not in (None, "") else None,
                    override_fx=float(override_fx) if override_fx not in (None, "") else None,
                    custom_text=custom_text,
                    uploaded_bytes=uploaded_bytes,
                    uploaded_filename=uploaded_filename,
                )
                self._send_json(result)
                return

            if route == "/api/explain":
                sentence = body.get("sentence", "")
                as_of = body.get("as_of", "2026-10-16")
                if not str(sentence).strip():
                    self._send_json({"error": "Please provide a non-empty sentence."}, status=HTTPStatus.BAD_REQUEST)
                    return
                explained = engine.explain_sentence(sentence)
                snapshots = engine.get_indicator_snapshot(as_of=as_of)
                is_assump = (
                    explained["rules_v1"]["is_assumption"] == "1"
                    or explained["model_a"]["probability"] >= 0.45
                )
                topic = explained["rules_v1"]["topic"] or "other"
                explained["claim"] = engine.evaluate_claim(sentence, topic, is_assump, snapshots)
                self._send_json(explained)
                return

            if route == "/api/review":
                sentence_id = body.get("sentence_id", "")
                doc_id = body.get("doc_id", "plan1")
                sentence = body.get("sentence", "")
                is_assumption = str(body.get("is_assumption", ""))
                topic = str(body.get("topic", ""))
                review_status = str(body.get("review_status", "reviewed"))
                if not sentence_id:
                    self._send_json({"error": "sentence_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                saved = engine.save_review_decision(
                    sentence_id=sentence_id,
                    doc_id=doc_id,
                    sentence=sentence,
                    is_assumption=is_assumption,
                    topic=topic,
                    review_status=review_status,
                )
                self._send_json({"status": "saved", "review": saved})
                return

            if route == "/api/add-assumption":
                doc_source = body.get("doc_source", "plan1")
                sentence = body.get("sentence", "")
                section = body.get("section", "5. Custom Strategic Assumptions")
                as_of = body.get("as_of", "2026-10-16")
                if not str(sentence).strip():
                    self._send_json({"error": "Please enter an assumption sentence."}, status=HTTPStatus.BAD_REQUEST)
                    return
                added = engine.add_assumption_clause(
                    doc_source=doc_source,
                    sentence=sentence,
                    section=section,
                    as_of=as_of,
                )
                self._send_json(added)
                return

            if route == "/api/generate-pdf":
                company_name = body.get("company_name", "AeroVantage Freight & Aviation Pvt. Ltd.")
                crude_low = float(body.get("crude_low", 68.0))
                crude_high = float(body.get("crude_high", 82.0))
                diesel_cap = float(body.get("diesel_cap", 94.0))
                fx_target = float(body.get("fx_target", 83.5))
                fx_trigger = float(body.get("fx_trigger", 87.5))
                rate_target = float(body.get("rate_target", 9.8))
                demand_growth = float(body.get("demand_growth", 11.0))
                as_of = body.get("as_of", "2026-10-16")
                mode = body.get("mode", "hybrid")
                threshold = float(body.get("threshold", 0.45))

                gen_info = engine.generate_sample_pdf(
                    company_name=company_name,
                    crude_low=crude_low,
                    crude_high=crude_high,
                    diesel_cap=diesel_cap,
                    fx_target=fx_target,
                    fx_trigger=fx_trigger,
                    rate_target=rate_target,
                    demand_growth=demand_growth,
                )
                analysis = engine.analyze_document(
                    doc_source="plan3",
                    as_of=as_of,
                    mode=mode,
                    threshold=threshold,
                )
                self._send_json({"generated": gen_info, "analysis": analysis})
                return

            if route == "/api/reset-session":
                doc_id = body.get("doc_id")
                engine.reset_reviews_and_clauses(doc_id=doc_id)
                self._send_json({"status": "reset", "doc_id": doc_id})
                return

            if route == "/api/retrain":
                lab_data = engine.get_model_lab_data(retrain=True)
                self._send_json(lab_data)
                return

            self.send_error(HTTPStatus.NOT_FOUND, f"Unknown POST route: {route}")
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def run_cli_demo():
    """Print a terminal walkthrough of the demo engine on the sample plans."""
    engine = get_engine()
    print("=" * 88)
    print("ASSUMPTION MONITOR — END-TO-END CLI DEMO REPORT")
    print("=" * 88)

    for doc_id in ("plan1", "plan2"):
        report = engine.analyze_document(doc_source=doc_id, as_of="2026-10-16", mode="hybrid")
        s = report["summary"]
        print(f"\nDocument: {report['doc_title']}  (as of {report['as_of']})")
        print(
            f"  Sentences: {s['total_sentences']} ({s['ok_sentences']} ok, {s['removed_lines']} cleaned/removed lines)"
        )
        print(
            f"  Assumptions detected: {s['assumptions_detected']} | "
            f"Monitored claims: {s['monitored_claims']} "
            f"(Breached: {s['breached_claims']}, Warning: {s['warning_claims']}, Holding: {s['holding_claims']})"
        )
        for c in report["monitored_claims"]:
            badge = c["status"].upper()
            print(f"    [{badge:8s}] ({c['sentence_id']}, {c['series_id']} {c['mapping_kind']}) {c['sentence']}")
            print(f"      -> {c['explanation']}")

    print("\n" + "=" * 88)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Assumption Monitor interactive web demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--cli", action="store_true", help="Run a terminal summary report and exit")
    args = parser.parse_args(argv)

    if args.cli:
        run_cli_demo()
        return 0

    get_engine()
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Assumption Monitor Demo UI running at: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
