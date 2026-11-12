from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanogate.config import Settings, load_dotenv


def _services(settings: Settings):
    from nanogate.api import build_services

    return build_services(settings)


def cmd_init(args, settings: Settings) -> int:
    services = _services(settings)
    target = Path(settings.db_path).parent / "bootstrap_keys.json"
    if target.exists() and not args.force:
        print(f"{target} already exists. Use --force to issue a new set.")
        return 1
    issued = {}
    for (tenant, department) in sorted(services.gateway.policy.departments):
        key_id, key = services.keys.create(tenant, department, "app", "bootstrap")
        issued[f"app:{tenant}/{department}"] = {"key_id": key_id, "key": key}
    for tenant in sorted({tenant for tenant, _ in services.gateway.policy.departments}):
        department = sorted(d for t, d in services.gateway.policy.departments if t == tenant)[0]
        for role in ("admin", "auditor"):
            key_id, key = services.keys.create(tenant, department, role, "bootstrap")
            issued[f"{role}:{tenant}"] = {"key_id": key_id, "key": key}
    target.unlink(missing_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(issued, handle, indent=2)
    print(f"Database ready at {settings.db_path}. Issued {len(issued)} keys to {target} (mode 0600).")
    return 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    from nanogate.api import create_app

    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port, log_level="info")
    return 0


def cmd_keys(args, settings: Settings) -> int:
    services = _services(settings)
    if args.action == "create":
        key_id, key = services.keys.create(args.tenant, args.department, args.role, args.label)
        print(json.dumps({"key_id": key_id, "key": key}))
    elif args.action == "revoke":
        if not services.keys.revoke(args.key_id):
            print("No active key with that id.")
            return 1
        print(f"Revoked {args.key_id}.")
    else:
        for row in services.keys.list():
            state = "revoked" if row["revoked_at"] else "active"
            print(f"{row['key_id']}  {row['tenant']}/{row['department']:<10} {row['role']:<8} {state:<8} {row['label']}")
    return 0


def cmd_verify(args, settings: Settings) -> int:
    services = _services(settings)
    expected = json.loads(Path(args.head).read_text(encoding="utf-8")) if args.head else None
    result = services.gateway.receipts.verify(expected_head=expected)
    print(json.dumps({**result, "head": services.gateway.receipts.head()}, indent=2))
    return 0 if result["status"] == "SEALED" else 2


def cmd_train(args, settings: Settings) -> int:
    from nanogate.router import load_rows, save, train

    artifact = train(load_rows(args.data), target_error=args.target_error, seed=args.seed)
    output = args.output or settings.router_artifact
    save(artifact, output)
    print(json.dumps({"saved": output, "threshold": artifact["threshold"], "rows": artifact["rows"], "held_out": artifact["held_out"]}, indent=2))
    return 0


def cmd_attack(args, settings: Settings) -> int:
    from nanogate.attack import run_suite

    report = run_suite(args.cases, settings)
    for failure in report["failures"]:
        print(f"FAIL {failure['id']}: {failure['why']}")
    print(f"{report['passed']}/{report['total']} attack cases held.")
    return 0 if not report["failures"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanogate", description="Local-first AI gateway.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create the database and issue bootstrap keys")
    init.add_argument("--force", action="store_true")
    init.set_defaults(run=cmd_init)

    serve = commands.add_parser("serve", help="run the gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(run=cmd_serve)

    keys = commands.add_parser("keys", help="manage API keys")
    actions = keys.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("tenant")
    create.add_argument("department")
    create.add_argument("--role", default="app", choices=("app", "auditor", "admin"))
    create.add_argument("--label", default="")
    revoke = actions.add_parser("revoke")
    revoke.add_argument("key_id")
    actions.add_parser("list")
    keys.set_defaults(run=cmd_keys)

    verify = commands.add_parser("verify", help="verify the receipt chain")
    verify.add_argument("--head", help="JSON file with an anchored {seq, digest} to compare against")
    verify.set_defaults(run=cmd_verify)

    train = commands.add_parser("train-router", help="fit the risk router from graded JSONL rows")
    train.add_argument("data")
    train.add_argument("--output")
    train.add_argument("--target-error", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(run=cmd_train)

    attack = commands.add_parser("attack", help="run the attack suite against an in-process gateway")
    attack.add_argument("--cases", default="bench/attacks.json")
    attack.set_defaults(run=cmd_attack)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.run(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
