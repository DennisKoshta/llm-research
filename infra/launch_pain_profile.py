"""Consume and execute one bounded method-control phase from its frozen copy."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import budget
import modal
from pain_profile_modal import create_app, artifacts
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.pain_profile_admission import claim, verify_frozen_campaign, CAMPAIGN_NAME
from curiosity_study.pain_profile_assets import require
from curiosity_study import pain_profile_resources as resources


def stop_app(cli, app_id):
    if app_id:
        subprocess.run([str(cli), "app", "stop", app_id], check=True, timeout=30,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def collect(run):
    """Export all observed bytes; refuse to overwrite any differing local record."""
    run = Path(run)
    count = 0
    for entry in artifacts.iterdir(CAMPAIGN_NAME, recursive=True):
        # FileEntryType.FILE has value 1 in the pinned local Modal SDK.
        if int(entry.type) != 1:
            continue
        name = Path(entry.path)
        if name.is_absolute():
            name = name.relative_to("/")
        require(name.parts[0] == CAMPAIGN_NAME and ".." not in name.parts, "Unsafe artifact path")
        destination = run / Path(*name.parts[1:])
        payload = b"".join(artifacts.read_file(entry.path))
        if destination.exists():
            require(destination.read_bytes() == payload, "Cloud artifact conflicts with local immutable file")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=list(resources.POLICIES), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest, frozen, root = verify_frozen_campaign(run)
    execution = Path(frozen["execution_root"])
    require(Path(__file__).resolve().parents[1] == execution, "Launch only the frozen script")
    budget.ROOT = root
    cli = root / ".venv/bin/modal"
    initial = budget.snapshot(timeout=20)
    admission = claim(run, args.phase, initial)
    prior_files = {}
    for name in ("phases/prefetch/summary.json", "phases/setup/summary.json", "phases/setup/audit.json"):
        if (run / name).is_file():
            prior_files[name] = read(run / name)
    app, job, image = create_app(execution, args.phase)
    deadline = admission["deadline"]
    done = threading.Event()
    stop_errors = []

    def watchdog():
        # Includes image startup, not just the remote function. Do not touch
        # unrelated apps, even if another user has one running in the account.
        while not done.wait(max(0.1, min(2.0, deadline - time.time()))):
            if time.time() < deadline:
                continue
            try:
                apps = json.loads(subprocess.check_output([str(cli), "app", "list", "--json"],
                                                          text=True, timeout=20))
                for candidate in apps:
                    created = dt.datetime.fromisoformat(candidate["Created at"]).timestamp()
                    if (candidate.get("Description") == admission["policy"]["app_name"]
                            and candidate["State"] != "stopped" and created >= admission["created"] - 2):
                        stop_app(cli, candidate["App ID"])
            except Exception as error:
                stop_errors.append({"type": type(error).__name__, "message": str(error)})

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    call, app_id, result, failure = None, None, None, None
    try:
        with modal.enable_output(), app.run():
            app_id = app.app_id
            write_once(run / "launches" / f"{args.phase}.json",
                       dict(app_id=app_id, admission_sha256=digest(admission), manifest_sha256=digest(manifest),
                            modal_image_id=image.object_id, started=time.time(), deadline=deadline))
            require(time.time() < deadline - 60, "Startup consumed the phase wall")
            call = job.spawn(manifest, admission, prior_files, image.object_id)
            while result is None:
                remaining = deadline - time.time()
                require(remaining > 0, "Phase reached fixed wall deadline")
                try:
                    result = call.get(timeout=min(15, remaining))
                except TimeoutError:
                    continue
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        app_id = app_id or app.app_id
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as error:
                stop_errors.append({"type": type(error).__name__, "message": str(error)})
        try:
            stop_app(cli, app_id)
        except Exception as error:
            stop_errors.append({"type": type(error).__name__, "message": str(error)})
        # Verify shutdown within the separately reserved 120-second teardown.
        shutdown_deadline = time.time() + 110
        state = None
        shutdown_errors = []
        while state is None or not state["all_stopped_zero_tasks"]:
            try:
                state = budget.snapshot(timeout=20)
            except Exception as error:
                shutdown_errors.append({"type": type(error).__name__, "message": str(error)})
            if (state is not None and state["all_stopped_zero_tasks"]) or time.time() >= shutdown_deadline:
                break
            time.sleep(3)
        if state is None:
            state = dict(initial, all_stopped_zero_tasks=False, shutdown_verified=False,
                         shutdown_verification_errors=shutdown_errors,
                         timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            if app_id:
                unresolved = {"App ID": app_id, "Description": admission["policy"]["app_name"],
                              "Created at": dt.datetime.fromtimestamp(admission["created"], dt.timezone.utc).isoformat(),
                              "Stopped at": None, "State": "unverified", "Tasks": "unknown"}
                state["apps"] = [*initial["apps"], unresolved]
            state["caveat"] = "Shutdown and new phase cost unverified; reconcile before further admission."
        done.set()
        watcher.join(timeout=25)
        write_once(root / "reports" / f"{CAMPAIGN_NAME}-{args.phase}-cost-and-shutdown.json", state)
        exported, export_error = None, None
        try:
            exported = collect(run)
        except Exception as error:
            export_error = {"type": type(error).__name__, "message": str(error)}
        outcome = dict(phase=args.phase, app_id=app_id, result=result, failure=failure,
                       stop_errors=stop_errors, all_stopped_zero_tasks=state["all_stopped_zero_tasks"],
                       shutdown_verification_errors=shutdown_errors,
                       exported_files=exported, export_error=export_error,
                       admission_sha256=digest(admission), manifest_sha256=digest(manifest),
                       finished=time.time())
        write_once(run / "launches" / f"{args.phase}-outcome.json", outcome)
        print(json.dumps({key: value for key, value in outcome.items() if key != "result"}, indent=2), flush=True)
    if failure or export_error or not state["all_stopped_zero_tasks"] or not result or not result["passed"]:
        raise SystemExit(1)
    summary = run / "phases" / args.phase / "summary.json"
    print(json.dumps({"phase": args.phase, "passed": True,
                      "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
